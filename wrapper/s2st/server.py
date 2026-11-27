"""Standard engine entry point for the hash-locked composite S2ST payloads."""
import importlib
import os

from ..contract import EngineArgs, register
from ..gpu import mount_metrics


def build_app(supports):
    base = os.environ.get('AUDIO_BASE')
    if base not in ('s2st-tencent', 's2st-youdao'):
        raise ValueError('Unknown locked S2ST image family')
    family = base.removeprefix('s2st-')
    module = importlib.import_module('wrapper.s2st.locked_payload.' + family + '.api.api')
    app = module.build_app(family, EngineArgs(), supports)
    register(app, model_name=os.environ.get('MODEL_NAME', base), module='s2st', served=supports, task_api=True,
             is_ready=app.state.locked_is_ready, error=lambda: app.state.locked_status['fault'],
             repo=os.environ.get('MODEL_REPO'), sample_rate=24000)
    mount_metrics(app)
    return app


def main(supports):
    import uvicorn
    uvicorn.run(build_app(supports), host='0.0.0.0', port=int(os.getenv('ENGINE_PORT', '8000')),
                ws_ping_interval=None, ws_ping_timeout=None)
