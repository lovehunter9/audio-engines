"""Original SimulInfer app/session, with only its model-call transport replaced.

The author GGUF implementation lacks a model server. This adapter preserves
the original app's session settings, VAD, WAIT, prompt/history, ST and EOF.
"""
import argparse
import asyncio
import dataclasses
import hashlib
import json
from pathlib import Path

import uvicorn

ROOT = Path('/home/olares/s2st-original-check-20261123')


async def serve(options):
    from simulinfer.config import load
    from simulinfer import app, translation
    from tencent_revised_rpc import NativeAPI, NativePool

    settings = load(options.config)
    # The native GGUF program has no HTTP /models endpoint. Preserve the
    import httpx
    from simulinfer.model_discovery import discover
    async def model_metadata(request):
        if not request.url.path.endswith('/models'):
            raise RuntimeError('Unexpected metadata request: ' + str(request.url))
        return httpx.Response(200, json={'data': [{
            'id': 'official-Q4_K_M',
            'root': str(ROOT / 'models/Hy-SimulMT-30B-A3B'),
        }]})
    if settings.model_endpoints:
        async with httpx.AsyncClient(transport=httpx.MockTransport(model_metadata)) as http:
            settings = await discover(settings, http=http)

    if settings.max_sessions != 1:
        raise ValueError('One shared quantized backend requires one native session at a time')
    if not settings.no_chinese_target or not settings.model_logits_processors:
        raise ValueError('Validated official target-language policy must be explicitly enabled')
    root = ROOT / 'sources/quantized-003-snapshot'
    binary = ROOT / 'results/conformant-tencent-native-20261125-r4/llama-mtmd-native-rpc'
    pool = NativePool(root, options.output / 'resident-backend', binary, settings.no_chinese_target)
    try:
        await pool.start()
        if options.activation_controls:
            from small_controls import check_tencent_mask
            await check_tencent_mask(pool, options.output/'activation-controls.json')
    except BaseException:
        await pool.close()
        raise
    counter = 0

    class QuantizedAPI(NativeAPI):
        def __init__(self, model, selected, mode, direction, http=None):
            nonlocal counter
            if mode != 'target_only' or selected.st_context:
                raise ValueError('Only this frozen translation-only workflow is validated')
            counter += 1
            directory = options.output / ('session-%05d' % counter)
            directory.mkdir()
            super().__init__(pool, selected, directory, direction)
            self.initialized = False

        async def generate(self, *args, **kwargs):
            if not self.initialized:
                await pool.reset()
                self.initialized = True
            return await super().generate(*args, **kwargs)

    translation.ChatAPI = QuantizedAPI
    source = Path(translation.__file__)
    specials = {}
    for piece in ['<｜hy_EOT｜>', '<｜hy_end▁of▁sentence｜>',
                  '<｜hy_place▁holder▁no▁790｜>', '<｜hy_place▁holder▁no▁791｜>']:
        result = await pool.request({'operation': 'tokenize', 'text': piece})
        if len(result['tokens']) != 1:
            raise RuntimeError('Quantized tokenizer lacks an original special token')
        specials[piece] = result['tokens'][0]
    receipt = {
        'model_transport': 'Native GGUF calls; not the unsupported upstream llama-server',
        'original_translation_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'original_app_sha256': hashlib.sha256(Path(app.__file__).read_bytes()).hexdigest(),
        'settings': dataclasses.asdict(settings), 'special_tokens': specials,
        'unchanged': ['official app/session-settings', 'VAD configuration/CPU placement',
                      'source/frame schedule', 'WAIT/ST/EOF', 'prompt render and history',
                      'native greedy and no-repeat3 policy', 'external frozen MING'],
        'compatibility': ['Replace ChatAPI model transport only',
                          'Original model discovery uses local metadata transport because native GGUF has no HTTP /models',
                          'Reuse native GGUF text/audio encoder and weights',
                          'ST complete-item mel grouping matches BF16 processing.py',
                          'shared model, independent stream/ST KV sequences'],
        'startup_check': 'Original HTTP-vLLM protocol probes do not apply to a pipe; native special-token checks above, all model errors fail closed',
        'native_http_preflight_executed': False,
    }
    (options.output / 'transport-receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    server = uvicorn.Server(uvicorn.Config(app.build_app(settings, preflight_result=receipt),
                            host='127.0.0.1', port=settings.port, ws_ping_interval=None))
    try:
        await server.serve()
    finally:
        await pool.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--activation-controls', action='store_true')
    options = parser.parse_args()
    if not options.output.resolve().is_relative_to(ROOT.resolve() / 'runs'):
        raise ValueError('Output outside owned experiment root')
    options.output.mkdir(exist_ok=False)
    asyncio.run(serve(options))


if __name__ == '__main__':
    main()
