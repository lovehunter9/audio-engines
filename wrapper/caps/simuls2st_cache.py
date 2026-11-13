"""Keep streaming translation history aligned with the Thinker KV cache."""


def install(inference):
    import torch

    generate = inference.thinker.generate
    stream = inference.generate_streaming_semantic_codes
    prompt = None

    def generate_with_history(*args, **kwargs):
        if prompt is None or kwargs.get("past_key_values") is None:
            return generate(*args, **kwargs)
        step_ids = kwargs["input_ids"]
        kwargs["input_ids"] = prompt
        result = generate(*args, **kwargs)
        result.sequences = torch.cat((step_ids, result.sequences[:, prompt.shape[1]:]), dim=1)
        return result

    @torch.inference_mode()
    def stream_with_cache(**kwargs):
        nonlocal prompt

        history = [token for chunk in kwargs["history_chunks"]
                   for token in chunk["thinker_token_ids"]]
        prompt = torch.tensor(
            [history + list(kwargs["current_chunk_thinker_prefix_ids"])],
            device=inference.device, dtype=torch.long,
        )
        try:
            result = list(stream(**kwargs))
        finally:
            prompt = None
        cache = result[3]
        if cache is not None:
            full_ids = history + list(result[2]["thinker_token_ids"])
            seen = int(cache.get_seq_length())
            if seen > len(full_ids):
                raise RuntimeError("streaming translation cache exceeds its recorded history")
            if seen < len(full_ids):
                output = inference.thinker(
                    input_ids=torch.tensor([full_ids[seen:]], device=inference.device,
                                           dtype=torch.long),
                    attention_mask=torch.ones((1, len(full_ids)), device=inference.device,
                                              dtype=torch.long),
                    past_key_values=cache,
                    cache_position=torch.arange(seen, len(full_ids), device=inference.device,
                                                dtype=torch.long),
                    use_cache=True,
                    return_dict=True,
                )
                result[3] = output.past_key_values
                if int(result[3].get_seq_length()) != len(full_ids):
                    raise RuntimeError("streaming translation cache did not consume its reply suffix")
        return tuple(result)

    inference.thinker.generate = generate_with_history
    inference.generate_streaming_semantic_codes = stream_with_cache
