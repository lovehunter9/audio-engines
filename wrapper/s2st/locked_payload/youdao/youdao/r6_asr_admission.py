"""Schedule existing bound ASR/VAD methods; arguments and results pass unchanged."""
import functools

from r6_gpu_queue import configure, stage


def protect_models(asr, vad, queue):
    configure(queue, shared_upstream=True)
    for model, names in [(asr, ['streaming_transcribe', 'finish_streaming_transcribe']),
                         (vad, ['detect_chunk'])]:
        for name in names:
            original = getattr(model, name)

            @functools.wraps(original)
            def admitted(*args, _original=original, **kwargs):
                with stage('r2t2'):
                    return _original(*args, **kwargs)

            setattr(model, name, admitted)
