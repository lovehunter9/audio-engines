"""Native GGUF transport for the unchanged pinned SimulInfer VLLMSession.

No chunk, VAD, WAIT, recovery, crop or reset decision is implemented here.
Original budget.render supplies the exact prompt. Stream/ST share weights and
have separate CUDA KV sequences, so a recovery cannot destroy stream history.
"""
import asyncio
import base64
import copy
import hashlib
import json
import os
from pathlib import Path

EOT='<｜hy_EOT｜>'
AUDIO='<｜hy_place▁holder▁no▁6｜><｜hy_place▁holder▁no▁5｜><｜hy_place▁holder▁no▁8｜>'
MEDIA='<__media__>'


class NativePool:
    def __init__(self, root, directory, binary=None, english_target_mask=False):
        self.root=Path(root)
        self.directory=Path(directory)
        self.directory.mkdir(parents=True,exist_ok=False)
        self.binary=Path(binary) if binary else self.root/'results/host-probes-20261123/native-rpc-build2/llama-mtmd-native-rpc'
        self.english_target_mask=english_target_mask
        self.process=None
        self.reader=None
        self.log=None
        self.lock=asyncio.Lock()
        self.sequence=0
        self.tokenizer=None
        self.features=None
        self.events=(self.directory/'rpc.jsonl').open('w',buffering=1)

    async def start(self):
        env=os.environ.copy()
        env.pop('MTMD_TEST_RESPONSE_MARKER',None)
        env.update(json.loads((self.root/'results/runtime-env.json').read_text()))
        for key in list(env):
            if key.startswith('LLAMA_ARG_'):env.pop(key)
        from official_audio_features import OfficialFeatureCache
        self.features=OfficialFeatureCache(self.directory/'official-features',
            '999a35f26a01db13d865c375c2aee98f17cafe78d4e77848af49c8699ad79569')
        env['S2ST_OFFICIAL_MEL_CACHE']=str(self.features.directory)
        read_fd,write_fd=os.pipe()
        env.update(S2ST_NATIVE_RPC='1',S2ST_RPC_OUTPUT_FD=str(write_fd),
                   LD_LIBRARY_PATH=str(self.binary.parent)+':'+str(self.root/'sources/llama.cpp/build/bin')+':'+env.get('LD_LIBRARY_PATH',''))
        command=[str(self.binary),'-m',str(self.root/'models/Hy-SimulMT-30B-A3B-Q4_K_M.gguf'),
                 '-c','16384','-ngl','999','--temp','0','--seed','42','-fa','on',
                 '-t','12','-b','256','-ub','128','-ctk','q8_0','-ctv','q8_0',
                 '--top-k','0','--top-p','1','--min-p','0',
                 '--override-tensor','token_embd.weight=CUDA0']
        (self.directory/'configuration.json').write_text(json.dumps({'command':command,
            'binary_sha256':hashlib.sha256(self.binary.read_bytes()).hexdigest(),
            'weights':'Official Q4_K_M unchanged; unified16384KV capacity; q8KV retained from24GB feasibility','frontend':'Unmodified installed SimulInfer VLLMSession and budget.render',
            'backend_adaptations':['Internal JSON pipe only','Two unified CUDA KV sequences with one shared model',
                                   'Official ST whole-utterance audio item feature extraction',
                                   'Request-specific repetition penalty and no-repeat3'],
            'source_rewrite':False,'reference_supplied':False,'adoption':'matched quantized experiment; user authorized20261124'},indent=2)+'\n')
        # This is the exact HF tokenizer used by the sealed vLLM baseline.
        from transformers import AutoTokenizer
        self.tokenizer=AutoTokenizer.from_pretrained(
            '/home/olares/s2st-original-check-20261123/models/Hy-SimulMT-30B-A3B',
            local_files_only=True,trust_remote_code=False)
        self.log=(self.directory/'backend.log').open('wb')
        try:
            self.process=await asyncio.create_subprocess_exec(*command,env=env,stdin=asyncio.subprocess.PIPE,
                stdout=self.log,stderr=asyncio.subprocess.STDOUT,pass_fds=(write_fd,))
        finally:
            os.close(write_fd)
        self.reader=os.fdopen(read_fd,'r')
        ready=await asyncio.wait_for(asyncio.to_thread(self.reader.readline),240)
        if not ready:raise RuntimeError('Native backend exited before ready')
        data=json.loads(ready)
        if not data.get('ready'):raise RuntimeError(str(data))
        self.events.write(json.dumps({'ready':data})+'\n')
        model_root=Path('/home/olares/s2st-original-check-20261123/models/Hy-SimulMT-30B-A3B')
        generation=json.loads((model_root/'generation_config.json').read_text())
        model_stops=generation['eos_token_id']
        if isinstance(model_stops,int):model_stops=[model_stops]
        stops=set(model_stops)
        for literal in [EOT,'<｜hy_end▁of▁sentence｜>']:
            ids=self.tokenizer.encode(literal,add_special_tokens=False)
            if len(ids)!=1:raise RuntimeError('Non-atomic official stop string')
            stops.add(ids[0])
        accepted=await self.request({'operation':'official_stop_tokens','token_ids':sorted(stops)})
        if accepted.get('official_stop_tokens')!=sorted(stops):raise RuntimeError('Official stop policy not active')
        (self.directory/'official-stop-policy.json').write_text(json.dumps({
            'model_generation_eos_ids':model_stops,'explicit_request_stop_strings':[EOT,'<｜hy_end▁of▁sentence｜>'],
            'effective_stop_ids':sorted(stops),'source':'Same generation_config plus ChatAPI stop strings as the sealed BF16 service'},indent=2)+'\n')
        if self.english_target_mask:
            import ast,re
            import simulinfer
            path=Path(simulinfer.__file__).parent/'target_language.py'
            source=path.read_text()
            module=ast.parse(source)
            selected=[x for x in module.body if
                isinstance(x,ast.FunctionDef) and x.name in ['byte_decoder','blocked_token_ids'] or
                isinstance(x,ast.Assign) and any(isinstance(y,ast.Name) and y.id=='HAN' for y in x.targets)]
            scope={'re':re}
            exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),scope)
            native=await self.request({'operation':'vocabulary'})
            official_root=Path('/home/olares/s2st-original-check-20261123/models/Hy-SimulMT-30B-A3B')
            tokenizer=json.loads((official_root/'tokenizer.json').read_text())
            native_ids={int(value):token for token,value in native['model']['vocab'].items()}
            native_ids.update({int(item['id']):item['content'] for item in native['added_tokens']})
            expected_ids={int(value):token for token,value in tokenizer['model']['vocab'].items()}
            expected_ids.update({int(item['id']):item['content'] for item in tokenizer['added_tokens']})
            if any(native_ids.get(token_id)!=token for token_id,token in expected_ids.items()):
                raise RuntimeError('GGUF vocabulary differs from the exact official tokenizer IDs')
            blocked=set(scope['blocked_token_ids'](tokenizer))
            special={x['content']:x['id'] for x in tokenizer['added_tokens']}
            manifest_path=official_root/'export_manifest.json'
            manifest=json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
            fmt=manifest.get('source_target_format',{})
            for field,fallback in [('source_token','<｜hy_place▁holder▁no▁790｜>'),
                                   ('target_token','<｜hy_place▁holder▁no▁791｜>')]:
                token_id=special.get(fmt.get(field),special.get(fallback))
                if token_id is not None:blocked.add(token_id)
            generation=json.loads((official_root/'generation_config.json').read_text())
            if set(generation['eos_token_id']) & blocked:
                raise RuntimeError('The official target mask unexpectedly blocks an EOG token')
            await self.request({'operation':'target_token_mask','token_ids':sorted(blocked)})
            (self.directory/'official-target-mask.json').write_text(json.dumps({
                'official_source':str(path),'official_source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'functions_reused_unchanged':['byte_decoder','blocked_token_ids','HAN'],
                'mask_sha256':hashlib.sha256(json.dumps(sorted(blocked)).encode()).hexdigest(),
                'blocked_token_count':len(blocked),'scope':'zh-en targetonly; no post-hoc deletion',
                'official_tokenizer_sha256':hashlib.sha256((official_root/'tokenizer.json').read_bytes()).hexdigest(),
                'all_official_vocabulary_IDs_match_native':True,
                'stream_and_ST_requests_use_same_direction_policy':True},indent=2)+'\n')

    async def request(self, value):
        async with self.lock:
            self.sequence+=1
            self.events.write(json.dumps({'request':self.sequence,'value':value},ensure_ascii=False)+'\n')
            self.process.stdin.write((json.dumps(value,ensure_ascii=False)+'\n').encode())
            await self.process.stdin.drain()
            raw=await asyncio.wait_for(asyncio.to_thread(self.reader.readline),180)
            if not raw:raise RuntimeError('Native backend closed during request')
            result=json.loads(raw)
            self.events.write(json.dumps({'response':self.sequence,'value':result},ensure_ascii=False)+'\n')
            if 'error' in result:raise RuntimeError(result['error'])
            return result

    async def reset(self):
        for sequence in [0,1]:
            await self.request({'operation':'reset','sequence':sequence,'reset':True})

    async def close(self):
        if self.process and self.process.returncode is None:
            try:
                self.process.stdin.write(b'{"operation":"close"}\n')
                await self.process.stdin.drain()
                await asyncio.wait_for(self.process.wait(),10)
            except (TimeoutError,BrokenPipeError,ConnectionResetError):
                self.process.terminate()
                await self.process.wait()
        if self.reader:self.reader.close()
        if self.log:self.log.close()
        self.events.close()
        if self.features:
            (self.directory/'official-feature-receipt.json').write_text(json.dumps(self.features.receipt(),indent=2)+'\n')
            self.features.clear()
            self.features=None


class NativeAPI:
    def __init__(self,pool,settings,directory,direction=None):
        self.pool=pool
        self.settings=settings
        self.direction=direction
        self.directory=Path(directory)/'native-inputs'
        self.directory.mkdir(exist_ok=False)
        self.messages=[]
        self.last_prompt=''
        self.last_prefix=''

    def audio_paths(self,messages):
        paths=[]
        for message in messages:
            if message['role']!='user':continue
            for part in message['content']:
                if part['type']!='input_audio':continue
                audio=part['input_audio']
                if audio['format']!='wav':raise ValueError('Only original WAV audio parts supported')
                data=base64.b64decode(audio['data'],validate=True)
                name=hashlib.sha256(data).hexdigest()+'.wav'
                path=self.directory/name
                if not path.exists():path.write_bytes(data)
                if self.pool.features is None:raise RuntimeError('Official audio feature provider not initialized')
                self.pool.features.ensure(path)
                paths.append(str(path))
        return paths

    async def tokenize(self,text):
        native=await self.pool.request({'operation':'tokenize','text':text})
        reference=self.pool.tokenizer.encode(text,add_special_tokens=False)
        if native['tokens']!=reference:
            raise RuntimeError('Native text tokenization differs from the sealed HF tokenizer')
        return native

    async def canonical_prompt(self,messages,full):
        from simulinfer.budget import speech_tokens
        tokens=(await self.tokenize(full))['tokens']
        marker=self.pool.tokenizer.encode('<｜hy_place▁holder▁no▁5｜>',add_special_tokens=False)
        if len(marker)!=1:raise RuntimeError('Non-atomic official audio marker')
        counts=iter(speech_tokens(part.num_samples) for message in messages if message['role']=='user'
                    for part in message['content'] if part['type']=='input_audio')
        expanded=[]
        for token in tokens:
            if token==marker[0]:
                count=next(counts,None)
                if count is None:raise RuntimeError('Unmatched canonical audio marker')
                expanded.extend([token]*count)
            else:expanded.append(token)
        if next(counts,None) is not None:raise RuntimeError('Missing canonical audio marker')
        return expanded

    async def budget(self,messages,*,prefix=None):
        from simulinfer.budget import render,PromptBudget
        text,expansion,items=render(messages,'target_only',prefix)
        tokens=await self.tokenize(text)
        return PromptBudget(tokens['count']+expansion,self.settings.max_model_len,items)

    async def generate(self,messages,*,prefix=None,max_tokens=None,target_only=False,
                       repetition_penalty=None,source_phase=False,st=False,budget=None):
        from simulinfer.budget import render,ContextBudgetError
        from simulinfer.translation import Reply,split_reply
        if source_phase or prefix is not None:
            raise ValueError('This frozen target_only/st_context=false experiment cannot enable continuation modes')
        maximum=max_tokens or self.settings.max_tokens
        if self.settings.token_budget_enabled:
            budget=budget or await self.budget(messages,prefix=prefix)
        if budget and not budget.fits(maximum):raise ContextBudgetError('Native context budget exceeded')
        full,_,_=render(messages,'target_only',prefix)
        value={'sequence':1 if st else 0,'max_tokens':maximum,
               'english_target_mask':bool(self.settings.no_chinese_target and self.direction=='zh-en'),
               'repetition_penalty':self.settings.repetition_penalty if repetition_penalty is None else repetition_penalty,
               'no_repeat_ngram_size':self.settings.st_no_repeat_ngram_size if st else self.settings.no_repeat_ngram_size}
        if st or len(messages)==1:
            value['reset']=True
            selected=messages
            prompt=full
        elif full.startswith(self.last_prefix):
            # vLLM re-renders assistant text and terminates history with EOT.
            value['replace_last_assistant']=self.messages[-1]['content']
            selected=messages[-1:]
            prompt=full[len(self.last_prefix):]
        elif (len(messages[:-1])==len(self.messages) and messages[:-2]==self.messages[:-1]
              and messages[-2]['role']=='assistant'):
            # Original recovery commits by replacing only the final assistant.
            changed=messages[-2]['content']
            prior=self.last_prompt+changed+EOT
            if not full.startswith(prior):raise RuntimeError('Unexpected official assistant rewrite shape')
            value['replace_last_assistant']=changed
            selected=messages[-1:]
            prompt=full[len(prior):]
        else:
            raise RuntimeError('Unexpected official history mutation; no silent replay or omitted history')
        canonical=await self.canonical_prompt(messages,full)
        value['canonical_prompt_ids']=canonical
        value['full_prompt']=full.replace(AUDIO,MEDIA)
        value['full_audio_paths']=self.audio_paths(messages)
        if budget and len(canonical)!=budget.prompt_tokens:
            raise RuntimeError('Canonical prompt differs from original budget calculation')
        value['prompt']=prompt.replace(AUDIO,MEDIA)
        value['audio_paths']=self.audio_paths(selected)
        if value['prompt'].count(MEDIA)!=len(value['audio_paths']):
            raise RuntimeError('Exact official audio markers differ from native media population')
        result=await self.pool.request(value)
        if budget and budget.prompt_tokens!=result['prompt_tokens']:
            raise RuntimeError('Official prompt/audio token count differs from native backend')
        if result.get('canonical_prompt_verified') is not True:
            raise RuntimeError('Backend generated without exact canonical prompt verification')
        finish=result['finish_reason']
        if finish=='length' and not self.settings.reset_on_max_tokens:
            raise RuntimeError('Original output budget reached; do not accept a truncated translation')
        raw=result['text'].removesuffix(EOT).removesuffix('<｜hy_end▁of▁sentence｜>')
        source,text=('',raw) if target_only else split_reply(raw,'target_only')
        if not st:
            self.last_prompt=full
            self.last_prefix=full+raw+EOT
            self.messages=copy.deepcopy(messages)+[{'role':'assistant','content':raw}]
        return Reply(source,text,raw,result['token_ids'],result['stop'],
                     result['prompt_tokens'],result['completion_tokens'],finish)

    async def close(self):
        # VLLMSession owns its logical state; the cohort owns resident weights.
        if self.pool.features:self.pool.features.clear()
        return
