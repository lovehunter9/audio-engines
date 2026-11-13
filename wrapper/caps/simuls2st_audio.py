"""Align VoiceBox's acoustic prompt to its semantic-code prefix."""


def install(agent):
    import numpy as np
    import torch

    original = agent._synthesize_codes

    @torch.inference_mode()
    def synthesize(generated_codes, states, current_audio):
        if not agent.use_prompt_mel:
            return original(generated_codes, states, current_audio)
        if generated_codes.shape[1] == 0:
            return np.empty(0, dtype=np.float32)
        inference = agent.inference
        ref_16k, ref_24k = agent._build_flow_matching_prompt_audio(states, current_audio)
        ref_codes = torch.as_tensor(inference.audio_tokenizer(ref_16k.numpy()),
                                    device=agent.model_device, dtype=torch.long).reshape(1, -1)
        prompt = inference.extract_mel_feature(ref_24k.unsqueeze(0).to(agent.model_device))
        prefix_frames = ref_codes.shape[1] * int(inference.voicebox.cond_scale_factor)
        if prefix_frames == 0 or prompt.shape[1] == 0:
            raise RuntimeError("VoiceBox reference produced an empty conditioning prefix")
        if prompt.shape[1] < prefix_frames:
            tail = prompt[:, -1:].expand(-1, prefix_frames - prompt.shape[1], -1)
            prompt = torch.cat([prompt, tail], dim=1)
        else:
            prompt = prompt[:, :prefix_frames]
        mel = inference.codes_to_mel(
            generated_codes=generated_codes.to(device=agent.model_device, dtype=torch.long),
            prompt_codes=ref_codes, prompt_mel=prompt,
        )
        return inference.mel_to_audio(mel)

    agent._synthesize_codes = synthesize
