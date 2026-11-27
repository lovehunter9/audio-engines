"""External GPU admission; forward HTTP payload and response bytes unchanged."""
import argparse
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import httpx
import uvicorn

from r6_gpu_queue import async_stage, configure


def create_app(role, queue, upstream):
    configure(queue, shared_upstream=True)

    @asynccontextmanager
    async def lifespan(app):
        async with httpx.AsyncClient(base_url=upstream, timeout=180, trust_env=False) as client:
            app.state.client = client
            yield

    app = FastAPI(lifespan=lifespan)

    @app.api_route('/{path:path}', methods=['GET', 'POST'])
    async def forward(path: str, request: Request):
        body = await request.body()
        gate = async_stage(role, queue) if request.method == 'POST' else None
        if gate:
            await gate.__aenter__()
        response = None
        try:
            headers = {k: v for k, v in request.headers.items()
                       if k.lower() not in ('host', 'content-length', 'connection')}
            outgoing = app.state.client.build_request(request.method, '/' + path,
                content=body, headers=headers, params=request.query_params)
            response = await app.state.client.send(outgoing, stream=True)
        except BaseException:
            if gate:
                await gate.__aexit__(None, None, None)
            raise

        async def chunks():
            try:
                async for chunk in response.aiter_raw():
                    yield chunk
            finally:
                try:
                    await response.aclose()
                finally:
                    if gate:
                        await gate.__aexit__(None, None, None)

        return StreamingResponse(chunks(), status_code=response.status_code,
            headers={k: v for k, v in response.headers.items()
                     if k.lower() not in ('transfer-encoding', 'connection')})

    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', choices=['ming', 't3po'], required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--queue', required=True)
    parser.add_argument('--upstream', required=True)
    args = parser.parse_args()
    uvicorn.run(create_app(args.role, args.queue, args.upstream), host='127.0.0.1',
                port=args.port, log_level='warning')
