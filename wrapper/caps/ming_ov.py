# Ming-3.0-tts on the Intel GPU. Python only sequences; VAE, DiT, Qwen2 and campplus stay on GPU.
import logging
import os

log = logging.getLogger("audio-ming")

VAE_FRAME_BUCKETS = (64, 128, 256, 384, 512, 768, 1024, 1536, 2048)


def bucket_for(n, buckets):
    """Smallest bucket that holds n. Longer than the last bucket is a stop, not a CPU compile."""
    n = int(n)
    for bucket in buckets:
        if n <= bucket:
            return int(bucket)
    raise RuntimeError(
        "length %s is past the last OpenVINO bucket %s" % (n, buckets[-1]))


def _numpy(tensor):
    import numpy as np

    arr = tensor.detach().float().cpu().numpy() if hasattr(tensor, "detach") else tensor
    return np.ascontiguousarray(arr, dtype="float32")


def _torch(array, like):
    import torch

    out = torch.from_numpy(array)
    if hasattr(like, "dtype"):
        out = out.to(dtype=like.dtype)
    return out


def _hide_cpu_onnxruntime():
    """campplus.onnx is read by OpenVINO. The stub keeps a __spec__ so diffusers can import."""
    import importlib.machinery
    import sys
    import types

    if getattr(sys.modules.get("onnxruntime"), "_ming_ov_stub", False):
        return

    class _Session:
        def __init__(self, *args, **kwargs):
            self.path = args[0] if args else ""

        def run(self, *args, **kwargs):
            raise RuntimeError("campplus runs on the Intel GPU")

        def get_inputs(self):
            class _In:
                name = "x"
            return [_In()]

    class _Options:
        graph_optimization_level = None
        intra_op_num_threads = 1

    class _Level:
        ORT_ENABLE_ALL = 0

    fake = types.ModuleType("onnxruntime")
    fake.__spec__ = importlib.machinery.ModuleSpec("onnxruntime", loader=None)
    fake.InferenceSession = _Session
    fake.SessionOptions = _Options
    fake.GraphOptimizationLevel = _Level
    fake._ming_ov_stub = True
    sys.modules["onnxruntime"] = fake


def _silence_cuda_autocast():
    import contextlib
    import torch

    if getattr(torch.autocast, "_ming_ov", False):
        return
    real = torch.autocast

    def autocast(*args, **kwargs):
        device_type = kwargs.get("device_type", args[0] if args else None)
        if device_type == "cuda":
            return contextlib.nullcontext()
        return real(*args, **kwargs)

    autocast._ming_ov = True
    torch.autocast = autocast


def _props(device):
    from .. import tts_ov

    props, prec = tts_ov._device_props(device, "f32")
    if str(prec).lower() not in ("f32", "fp32"):
        raise RuntimeError("Ming OpenVINO stays fp32 on GPU, got %s" % prec)
    return props


def _saved_input_shapes(xml):
    import re

    text = open(xml, encoding="utf-8", errors="replace").read()
    found = re.findall(r'<layer\b[^>]*type="Parameter"[\s\S]*?<data shape="([^"]+)"', text)
    shapes = []
    for raw in found:
        dims = []
        for item in raw.split(","):
            item = item.strip()
            dims.append(-1 if item in ("?", "-1") else int(item))
        shapes.append(tuple(dims))
    return shapes


def _refuse_dynamic_spectral(model, xml):
    for op in model.get_ordered_ops():
        if op.get_type_name() not in ("IRDFT", "DFT", "IDFT", "RDFT", "Col2Im"):
            continue
        for out in op.outputs():
            shape = out.get_partial_shape()
            if not shape.is_static:
                raise RuntimeError(
                    "%s %s stayed dynamic (%s); refusing to compile it on GPU"
                    % (os.path.basename(xml), op.get_type_name(), shape))


def _export(mod, example, xml):
    """Pin every input to the example. A dynamic IRDFT is what SIGSEGV'd the GPU compiler."""
    import openvino as ov
    import torch

    from .. import tts_ov

    os.makedirs(os.path.dirname(xml), exist_ok=True)
    if not isinstance(example, (tuple, list)):
        example = (example,)
    want = [tuple(int(x) for x in t.shape) for t in example]
    if os.path.isfile(xml) and _saved_input_shapes(xml) == want:
        return
    if hasattr(mod, "eval"):
        mod.eval()
    tts_ov._freeze_for_export(mod)
    log.info("exporting %s example=%s", xml, want)
    with torch.inference_mode():
        ov_model = ov.convert_model(mod, example_input=example)
    mapping = {}
    for inp, shape in zip(ov_model.inputs, want):
        mapping[inp.get_any_name()] = list(shape)
    ov_model.reshape(mapping)
    ov_model.validate_nodes_and_infer_types()
    _refuse_dynamic_spectral(ov_model, xml)
    ov.save_model(ov_model, xml)


def _compile_xml(xml, device, props):
    from .. import tts_ov

    log.info("compiling %s on %s", xml, device)
    return tts_ov._compile_ir(xml, device, props)


def _call(compiled, *arrays):
    import numpy as np

    feed = {}
    for i, arr in enumerate(arrays):
        if hasattr(arr, "detach"):
            arr = _numpy(arr)
        feed[compiled.inputs[i]] = np.ascontiguousarray(arr)
    return compiled(feed)[0]


class _Graph:
    """One static shape. The first bucket is compiled at load; the rest compile on first use."""

    def __init__(self, xml, device, props, prepare=None):
        self.xml = xml
        self.device = device
        self.props = props
        self.compiled = None
        self.prepare = prepare

    def ensure(self):
        if self.compiled is None:
            if self.prepare is not None:
                self.prepare()
            self.compiled = _compile_xml(self.xml, self.device, self.props)
        return self.compiled

    def __call__(self, *arrays):
        return _call(self.ensure(), *arrays)


def _patch_forward(module, graph, n_in):
    def forward(*tensors):
        arrays = [_numpy(t) for t in tensors[:n_in]]
        return _torch(graph(*arrays), tensors[0])

    module.forward = forward


class _DecodeVAE(object):
    def __init__(self, audio):
        import torch.nn as nn

        class _M(nn.Module):
            def __init__(self):
                super().__init__()
                self.audio = audio

            def forward(self, latent):
                wave, _, _ = self.audio.decode(
                    latent, past_key_values=None, use_cache=False, last_chunk=True)
                return wave

        self.mod = _M()


class _EncodeVAE(object):
    def __init__(self, encoder):
        import torch.nn as nn

        class _M(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = encoder

            def forward(self, waveform):
                h, _ = self.encoder(waveform)
                return h

        self.mod = _M()


class _DitStep(object):
    def __init__(self, dit, patch):
        import torch.nn as nn

        class _M(nn.Module):
            def __init__(self):
                super().__init__()
                self.dit = dit
                self.patch = int(patch)

            def forward(self, x, t, c, hist):
                return self.dit.forward_with_cfg(x, t.reshape(()), c, hist, self.patch)

        self.mod = _M()


def _install_vae_decode(bailing, src, device, props):
    import torch

    latent_dim = int(bailing.latent_dim)
    wrapped = _DecodeVAE(bailing.audio)
    graphs = {}
    for index, frames in enumerate(VAE_FRAME_BUCKETS):
        xml = os.path.join(src, "openvino", "vae_decode_%s.xml" % frames)
        example = torch.zeros(1, frames, latent_dim)
        _export(wrapped.mod, example, xml)
        graphs[frames] = _Graph(xml, device, props)
        if index == 0:
            graphs[frames].ensure()
    probe = graphs[VAE_FRAME_BUCKETS[0]](torch.zeros(1, VAE_FRAME_BUCKETS[0], latent_dim))
    samples = int(probe.shape[-1])
    if samples <= 0:
        raise RuntimeError("VAE decode bucket produced no audio; ISTFT did not stay on GPU")
    per_frame = samples / float(VAE_FRAME_BUCKETS[0])

    def decode(latent, past_key_values=None, use_cache=False, stream_state=(None, None, None),
               last_chunk=False):
        frames = int(latent.shape[1])
        bucket = bucket_for(frames, VAE_FRAME_BUCKETS)
        if frames < bucket:
            latent = torch.nn.functional.pad(latent, (0, 0, 0, bucket - frames))
        wave = _torch(graphs[bucket](latent), latent)
        keep = int(round(wave.shape[-1] * frames / float(bucket)))
        return wave[..., :keep], stream_state, None

    bailing.audio.decode = decode
    log.info("VAE decode on %s buckets=%s samples_per_frame=%.3f", device, VAE_FRAME_BUCKETS, per_frame)


def _static_frames(encoder, waveform):
    """Same windows as unfold. OpenVINO rejects aten::unfold on this encoder."""
    import torch
    import torch.nn.functional as F

    hop = int(encoder.hop_size)
    width = int(encoder.input_dim)
    total = int(waveform.shape[-1])
    num_frames = (total + hop - 1) // hop
    waveform = F.pad(waveform, (0, (num_frames - 1) * hop + width - total))
    frames = (int(waveform.shape[-1]) - width) // hop + 1
    starts = torch.arange(frames, device=waveform.device) * hop
    cols = torch.arange(width, device=waveform.device)
    index = (starts.reshape(-1, 1) + cols.reshape(1, -1)).reshape(-1)
    return waveform.index_select(-1, index).reshape(waveform.shape[0], frames, width)


def _install_encoder(ming, bailing, src, device, props):
    import torch

    align = int(ming._pad_align())
    if align <= 0:
        raise RuntimeError("Ming prompt align is %s" % align)
    lengths = tuple(frames * align for frames in VAE_FRAME_BUCKETS)
    encoder = bailing.audio.encoder
    encoder.get_frames = lambda waveform, enc=encoder: _static_frames(enc, waveform)
    wrapped = _EncodeVAE(encoder)
    graphs = {}
    for samples in lengths:
        xml = os.path.join(src, "openvino", "vae_encode_%s.xml" % samples)

        def prepare(xml=xml, samples=samples):
            _export(wrapped.mod, torch.zeros(1, samples), xml)

        graphs[samples] = _Graph(xml, device, props, prepare)
    graphs[lengths[0]].ensure()
    with torch.inference_mode():
        probed, _ = encoder(torch.zeros(1, lengths[0]))
    frames = int(probed.shape[1])
    if frames <= 0 or lengths[0] % frames != 0:
        raise RuntimeError("VAE encoder probe %s samples -> %s frames" % (lengths[0], frames))
    samples_per_frame = lengths[0] // frames

    def forward(waveform):
        samples = int(waveform.shape[-1])
        bucket = bucket_for(samples, lengths)
        pad = bucket - samples
        if pad:
            waveform = torch.nn.functional.pad(waveform, (pad, 0))
        h = _torch(graphs[bucket](waveform), waveform)
        keep = samples // samples_per_frame
        return h[:, -keep:, :], waveform[:, -samples:].unsqueeze(1)

    encoder.forward = forward
    log.info("VAE encoder on %s", device)


def _install_dit(bailing, src, device, props):
    import torch

    dit = bailing.flowloss.cfm.model
    patch = int(bailing.patch_size)
    if patch != 1:
        raise RuntimeError("Ming OpenVINO DiT step expects patch_size 1, got %s" % patch)
    hist = int(bailing.history_patch_size)
    latent = int(bailing.latent_dim)
    hidden = int(bailing.model.config.hidden_size)
    wrapped = _DitStep(dit, patch)
    xml = os.path.join(src, "openvino", "dit_step.xml")
    example = (
        torch.zeros(1, patch, latent),
        torch.zeros(1),
        torch.zeros(1, 1, hidden),
        torch.zeros(1, hist, latent),
    )
    _export(wrapped.mod, example, xml)
    graph = _Graph(xml, device, props)
    graph.ensure()

    def forward_with_cfg(x, t, c, latent_history, patch_size):
        if int(patch_size) != patch:
            raise RuntimeError("DiT graph is patch %s, got %s" % (patch, patch_size))
        if t.ndim == 0:
            t = t.reshape(1)
        return _torch(graph(x, t, c, latent_history), x)

    dit.forward_with_cfg = forward_with_cfg
    log.info("DiT step on %s", device)


def _install_small(module, example, xml, device, props, label):
    _export(module, example, xml)
    graph = _Graph(xml, device, props)
    graph.ensure()
    _patch_forward(module, graph, 1)
    log.info("%s on %s", label, device)


def _install_qwen(bailing, src, device, props):
    """Same prefill and DeviceKvRunner as FireRed. ReadValue is what aborted this GPU."""
    import torch

    from .. import tts_ov

    del props
    qwen = bailing.model
    inner = qwen.model
    if getattr(inner, "layers", None) is None:
        raise RuntimeError("Ming Qwen2 has no layers to compile")
    n_layers, n_kv, head_dim = tts_ov.kv_meta(inner.config)
    hidden = int(inner.config.hidden_size)
    example_t = 16
    past = []
    for _ in range(n_layers):
        past.append(torch.zeros(1, n_kv, example_t, head_dim, dtype=torch.float32))
        past.append(torch.zeros(1, n_kv, example_t, head_dim, dtype=torch.float32))
    _, pre_xml, pre_stamp = tts_ov.ir_paths(src, "ming_qwen_prefill", ".ov-ming-prefill-v1")
    prefill = tts_ov.compile_causal(
        tts_ov.causal_kv_module(inner, False),
        (torch.zeros(1, example_t, hidden, dtype=torch.float32),
         torch.ones(1, example_t, dtype=torch.long)),
        pre_xml, pre_stamp, device,
        dynamize_ranks=(2, 3),
    )
    embeds_q1 = torch.zeros(1, 1, hidden, dtype=torch.float32)
    mask_q1 = torch.ones(1, example_t + 1, dtype=torch.long)
    pos_q1 = torch.arange(example_t, example_t + 1, dtype=torch.long).unsqueeze(0)
    _, xml, stamp = tts_ov.ir_paths(src, "ming_qwen_decode", ".ov-ming-decode-v1")
    compiled = tts_ov.compile_causal(
        tts_ov.causal_kv_module(inner, True, external_position=True),
        (embeds_q1, mask_q1, pos_q1, *past), xml, stamp, device,
        dynamize_ranks=(2, 4),
    )
    runner = tts_ov.DeviceKvRunner(
        compiled, n_layers, prefill=prefill, external_position=True)
    try:
        runner.step(torch.zeros(1, 8, hidden))
        runner.step(torch.zeros(1, 1, hidden))
        runner.reset()
        runner.step(torch.zeros(1, 32, hidden))
        runner.step(torch.zeros(1, 1, hidden))
    except Exception as exc:
        raise RuntimeError(
            "Ming Qwen2 prefill did not run at 8 and 32; stopping (%s)" % exc) from exc
    runner.reset()

    def forward(*args, **kwargs):
        embeds = kwargs.get("inputs_embeds")
        if embeds is None and args:
            embeds = args[0]
        if kwargs.get("past_key_values") is None:
            runner.reset()
        return _QwenOut(runner.step(embeds))

    qwen.forward = forward
    log.info("Qwen2 prefill+DeviceKvRunner on %s layers=%s", device, n_layers)


class _QwenOut:
    def __init__(self, hidden):
        self.hidden_states = (hidden,)
        self.past_key_values = ("ov",)



def _install_campplus(ming, device, props):
    import openvino as ov

    from ..ov_compile_lock import gpu_compile

    path = os.path.join(ming.local_path, "campplus.onnx")
    if not os.path.isfile(path):
        raise RuntimeError("campplus.onnx is missing at %s" % path)
    core = ov.Core()
    core.set_property({"ENABLE_MMAP": False})
    model = core.read_model(path)
    inp = model.inputs[0]
    shape = inp.get_partial_shape()
    if shape.rank.get_length() != 3:
        raise RuntimeError("campplus input rank is %s" % shape.rank)
    shape[1] = -1
    model.reshape({inp.get_any_name(): shape})
    cfg = dict(props)
    cfg["ENABLE_MMAP"] = False
    with gpu_compile(device):
        try:
            compiled = core.compile_model(model, device, cfg)
        except Exception as exc:
            raise RuntimeError("campplus did not compile on GPU; stopping (%s)" % exc) from exc
    input_name = compiled.inputs[0]

    def _extract(speech):
        import numpy as np
        import torch
        import torchaudio.compliance.kaldi as kaldi

        feat = kaldi.fbank(speech, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        arr = np.ascontiguousarray(feat.unsqueeze(0).float().numpy(), dtype="float32")
        out = compiled({input_name: arr})[0]
        return torch.tensor(np.asarray(out).reshape(1, -1))

    ming.spkemb_extractor._extract_spk_embedding = _extract
    log.info("campplus on %s", device)


def _release(bailing):
    from .. import tts_ov

    qwen = bailing.model
    tts_ov.release_parameters(getattr(qwen.model, "layers", None), "ming-qwen")
    tts_ov.release_parameters(getattr(qwen.model, "norm", None), "ming-qwen-norm")
    tts_ov.release_parameters(bailing.flowloss, "ming-dit")
    tts_ov.release_parameters(bailing.audio.decoder, "ming-vae-decode")
    tts_ov.release_parameters(bailing.audio.encoder, "ming-vae-encode")
    tts_ov.release_parameters(bailing.linear_proj_audio, "ming-agg")
    tts_ov.release_parameters(bailing.stop_head, "ming-stop")
    tts_ov.release_parameters(bailing.spk_head, "ming-spk")


def install(ming, device):
    import torch

    bailing = ming.model
    if getattr(bailing, "model_type", "") != "dense":
        raise RuntimeError("Ming OpenVINO is the 0.5B dense checkpoint, got %r" % bailing.model_type)
    bailing.float()
    if hasattr(bailing.model, "config"):
        bailing.model.config._attn_implementation = "eager"
    _silence_cuda_autocast()
    props = _props(device)
    src = ming.local_path
    _install_vae_decode(bailing, src, device, props)
    _install_dit(bailing, src, device, props)
    hidden = int(bailing.model.config.hidden_size)
    ov_dir = os.path.join(src, "openvino")
    _install_small(
        bailing.linear_proj_audio, torch.zeros(1, 1, int(bailing.latent_dim)),
        os.path.join(ov_dir, "aggregator.xml"), device, props, "aggregator")
    _install_small(
        bailing.stop_head, torch.zeros(1, 1, hidden),
        os.path.join(ov_dir, "stop_head.xml"), device, props, "stop head")
    _install_small(
        bailing.spk_head, torch.zeros(1, 192),
        os.path.join(ov_dir, "spk_head.xml"), device, props, "speaker head")
    _install_encoder(ming, bailing, src, device, props)
    _install_qwen(bailing, src, device, props)
    _install_campplus(ming, device, props)
    _release(bailing)


def load_model(repo):
    """Load Ming on CPU only to trace, then compile every graph for the Intel GPU."""
    from .. import tts_ov

    device = tts_ov.require_gpu()
    _hide_cpu_onnxruntime()
    restore = tts_ov.force_cpu_torch_device()
    try:
        from ming_tts import MingTTS

        ming = MingTTS(repo, device="cpu", text_normalization=False)
    finally:
        restore()
    install(ming, device)
    return ming
