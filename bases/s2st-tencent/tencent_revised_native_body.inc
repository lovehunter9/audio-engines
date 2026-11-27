// Bare-host transport adapter; the Python caller retains the original session.
static int run_native_rpc(mtmd_cli_context & ctx, const common_params & params) {
    using json = nlohmann::json;
    const char * descriptor = std::getenv("S2ST_RPC_OUTPUT_FD");
    if (!descriptor) return 1;
    FILE * output = fdopen(std::stoi(descriptor), "w");
    if (!output) return 1;
    auto respond = [&](const json & value) {
        const auto text = value.dump();
        std::fprintf(output, "%s\n", text.c_str());
        std::fflush(output);
    };
    const auto audio_marker=common_tokenize(ctx.lctx,"<｜hy_place▁holder▁no▁5｜>",false,true);
    if (audio_marker.size()!=1) throw std::runtime_error("non-atomic canonical audio marker");
    ctx.rpc_audio_token=audio_marker[0];
    llama_pos positions[2] = {0, 0};
    llama_pos assistant_begin[2] = {0, 0};
    std::vector<llama_token> blocked_target_tokens;
    std::vector<llama_token> official_stop_tokens;
    auto decode = [&](const llama_tokens & tokens) {
        for (size_t off = 0; off < tokens.size(); off += ctx.n_batch) {
            ctx.batch.clear();
            const size_t count = std::min(tokens.size()-off, (size_t) ctx.n_batch);
            for (size_t k=0; k<count; ++k) {
                ctx.batch.add(tokens[off+k], ctx.n_past++, ctx.active_seq, k+1==count);
            }
            if (llama_process(ctx.lctx, LLAMA_PROCESS_TYPE_DECODE, ctx.batch.get())) {
                throw std::runtime_error("native RPC token decode failed");
            }
            ctx.rpc_tokens[ctx.active_seq].insert(ctx.rpc_tokens[ctx.active_seq].end(),
                tokens.begin()+off,tokens.begin()+off+count);
        }
    };
    respond({{"ready", true}, {"weights_loaded_once", true}, {"unified_context", llama_n_ctx(ctx.lctx)}});
    std::string line;
    while (std::getline(std::cin, line) && !g_is_interrupted) {
        try {
            const json request=json::parse(line);
            const auto operation=request.value("operation", std::string("generate"));
            if (operation=="close") break;
            if (operation=="vocabulary") {
                json vocab=json::object();
                json added=json::array();
                for (llama_token id=0;id<llama_vocab_n_tokens(ctx.vocab);++id) {
                    const std::string token=llama_vocab_get_text(ctx.vocab,id);
                    if (llama_vocab_is_control(ctx.vocab,id)) {
                        added.push_back({{"id",id},{"content",token},{"special",true}});
                    } else {
                        vocab[token]=id;
                    }
                }
                respond({{"decoder",{{"type","ByteLevel"}}},{"model",{{"vocab",vocab}}},{"added_tokens",added}});
                continue;
            }
            if (operation=="target_token_mask") {
                blocked_target_tokens=request.at("token_ids").get<std::vector<llama_token>>();
                for (auto id:blocked_target_tokens) {
                    if (id<0 || id>=llama_vocab_n_tokens(ctx.vocab) || llama_vocab_is_eog(ctx.vocab,id)) {
                        throw std::runtime_error("invalid target token mask or blocked EOG");
                    }
                }
                respond({{"blocked_tokens",blocked_target_tokens.size()}});
                continue;
            }
            if (operation=="official_stop_tokens") {
                official_stop_tokens=request.at("token_ids").get<std::vector<llama_token>>();
                if (official_stop_tokens.empty()) throw std::runtime_error("missing official stop tokens");
                for (const auto id:official_stop_tokens) {
                    if (id<0 || id>=llama_vocab_n_tokens(ctx.vocab)) throw std::runtime_error("invalid official stop token");
                }
                respond({{"official_stop_tokens",official_stop_tokens}});
                continue;
            }
            if (operation=="tokenize") {
                const auto ids=common_tokenize(ctx.lctx, request.at("text").get<std::string>(), false, true);
                respond({{"tokens", ids}, {"count", ids.size()}});
                continue;
            }
            const int sequence=request.value("sequence", 0);
            if (sequence<0 || sequence>1) throw std::runtime_error("invalid sequence");
            ctx.active_seq=sequence;
            ctx.n_past=positions[sequence];
            if (request.value("reset",false)) {
                if (!llama_memory_seq_rm(llama_get_memory(ctx.lctx),sequence,0,-1)) {
                    throw std::runtime_error("native sequence reset failed");
                }
                ctx.n_past=0;
                ctx.rpc_tokens[sequence].clear();
            }
            if (operation=="reset") {
                positions[sequence]=ctx.n_past;
                respond({{"reset",true}});
                continue;
            }
            if (request.contains("replace_last_assistant")) {
                if (assistant_begin[sequence]<0 || assistant_begin[sequence]>ctx.n_past) {
                    throw std::runtime_error("invalid assistant rewrite boundary");
                }
                if (!llama_memory_seq_rm(llama_get_memory(ctx.lctx),sequence,assistant_begin[sequence],-1)) {
                    throw std::runtime_error("native assistant rewrite failed");
                }
                ctx.n_past=assistant_begin[sequence];
                ctx.rpc_tokens[sequence].resize(ctx.n_past);
                decode(common_tokenize(ctx.lctx, request.at("replace_last_assistant").get<std::string>()+"<｜hy_EOT｜>",false,true));
            }
            ctx.chat_history.clear();
            if (ctx.n_past) ctx.chat_history.emplace_back();
            ctx.bitmaps.entries.clear();
            auto sampling=params.sampling;
            sampling.penalty_repeat=request.value("repetition_penalty",1.0f);
            sampling.penalty_last_n=llama_n_ctx(ctx.lctx);
            common_sampler_free(ctx.smpl);
            ctx.smpl=common_sampler_init(ctx.model,sampling);
            if (sequence==1) setenv("S2ST_GGUF_ST_WHOLE_AUDIO","1",1);
            else unsetenv("S2ST_GGUF_ST_WHOLE_AUDIO");
            for (const auto & path:request.at("audio_paths")) {
                if (!ctx.load_media(path.get<std::string>())) throw std::runtime_error("native audio load failed");
            }
            common_chat_msg message;
            message.role="user";
            message.content=request.at("prompt").get<std::string>();
            if (eval_message(ctx,message)) throw std::runtime_error("native multimodal prefill failed");
            const auto canonical=request.at("canonical_prompt_ids").get<std::vector<llama_token>>();
            bool canonical_reprefill=false;
            if (ctx.rpc_tokens[sequence]!=canonical) {
                // No model output is generated from a noncanonical cache.
                // Fall back to exact full-history prefill, then recheck every ID.
                canonical_reprefill=true;
                if (!llama_memory_seq_rm(llama_get_memory(ctx.lctx),sequence,0,-1)) {
                    throw std::runtime_error("canonical fallback sequence reset failed");
                }
                ctx.n_past=0;
                ctx.rpc_tokens[sequence].clear();
                ctx.chat_history.clear();
                ctx.bitmaps.entries.clear();
                for (const auto & path:request.at("full_audio_paths")) {
                    if (!ctx.load_media(path.get<std::string>())) throw std::runtime_error("canonical audio load failed");
                }
                common_chat_msg full_message;
                full_message.role="user";
                full_message.content=request.at("full_prompt").get<std::string>();
                if (eval_message(ctx,full_message)) throw std::runtime_error("canonical full prefill failed");
            }
            if (ctx.rpc_tokens[sequence]!=canonical || ctx.n_past!=(llama_pos)canonical.size()) {
                throw std::runtime_error("actual prompt IDs differ from the official full rendered prompt");
            }
            // vLLM repetition penalties cover all prompt IDs, including audio
            // placeholders, and the current output. Seed the fresh sampler once.
            for (const auto id:canonical) common_sampler_accept(ctx.smpl,id,false);
            const auto prompt_tokens=ctx.n_past;
            assistant_begin[sequence]=ctx.n_past;
            if (official_stop_tokens.empty()) throw std::runtime_error("official stopping policy not initialized");
            const int maximum=request.at("max_tokens").get<int>();
            if (maximum<1 || maximum>4096) throw std::runtime_error("invalid output token budget");
            const int ngram=request.value("no_repeat_ngram_size",3);
            if (ngram!=0 && ngram!=3) throw std::runtime_error("unsupported ngram configuration");
            // Match the official per-request n-gram reset boundaries.
            std::vector<llama_token> boundaries;
            for (const auto & text : {"<｜hy_EOT｜>", "<｜hy_end▁of▁sentence｜>",
                                      "<｜hy_place▁holder▁no▁790｜>", "<｜hy_place▁holder▁no▁791｜>"}) {
                auto ids=common_tokenize(ctx.lctx,text,false,true);
                if (ids.size()!=1) throw std::runtime_error("official boundary token is not atomic");
                boundaries.push_back(ids[0]);
            }
            auto is_boundary=[&](llama_token id) {
                return std::find(boundaries.begin(),boundaries.end(),id)!=boundaries.end();
            };
            llama_tokens generated;
            std::string text;
            std::string stop;
            bool ended=false;
            for (int i=0;i<maximum && !g_is_interrupted;++i) {
                if (request.contains("logit_bias")) {
                    float * logits=llama_get_logits_ith(ctx.lctx,-1);
                    for (auto it=request.at("logit_bias").begin();it!=request.at("logit_bias").end();++it) {
                        const int id=std::stoi(it.key());
                        if (id<0 || id>=llama_vocab_n_tokens(ctx.vocab)) throw std::runtime_error("invalid activation-control bias ID");
                        logits[id]+=it.value().get<float>();
                    }
                }
                if (request.value("english_target_mask",false)) {
                    if (blocked_target_tokens.empty()) throw std::runtime_error("target token mask not initialized");
                    float * logits=llama_get_logits_ith(ctx.lctx,-1);
                    for (auto id:blocked_target_tokens) logits[id]=-INFINITY;
                }
                if (ngram==3 && generated.size()>=2) {
                    float * logits=llama_get_logits_ith(ctx.lctx,-1);
                    const auto n=generated.size();
                    size_t begin=0;
                    for (size_t j=0;j<n;++j) if (is_boundary(generated[j])) begin=j+1;
                    for (size_t j=begin;j+2<n && n-begin>=2;++j) {
                        if (!is_boundary(generated[j+2]) && generated[j]==generated[n-2] && generated[j+1]==generated[n-1]) {
                            logits[generated[j+2]]=-INFINITY;
                        }
                    }
                }
                const auto id=common_sampler_sample(ctx.smpl,ctx.lctx,-1);
                common_sampler_accept(ctx.smpl,id,true);
                generated.push_back(id);
                const auto piece=common_token_to_piece(ctx.lctx,id);
                decode(llama_tokens{id});
                if (std::find(official_stop_tokens.begin(),official_stop_tokens.end(),id)!=official_stop_tokens.end()) {
                    ended=true;stop=piece;break;
                }
                text+=piece;
            }
            positions[sequence]=ctx.n_past;
            respond({{"canonical_prompt_verified",true},{"canonical_reprefill",canonical_reprefill},
                     {"canonical_prompt_ids",canonical},
                     {"text",text},{"token_ids",generated},{"stop",stop},
                     {"prompt_tokens",prompt_tokens},{"completion_tokens",generated.size()},
                     {"finish_reason",ended?"stop":"length"},{"sequence",sequence}});
        } catch (const std::exception & error) {
            respond({{"error",error.what()}});
            std::fclose(output);
            return 1;
        }
    }
    std::fclose(output);
    return 0;
}
