FROM docker.io/python:3.12-slim-bookworm AS tempo
RUN python -m pip install --no-cache-dir --no-deps --target /overlay imageio-ffmpeg==0.8.1

# This immutable multi-arch parent has the verified official native binaries and MING stack.
# vendor.Dockerfile retains the complete source bootstrap recipe.
FROM docker.io/lovehunter9/audio-s2st-tencent:deps-7bc3825f4572@sha256:bca082f5414c0f583db29a2cc7efdcc28f06bb6edb59bb51d3bf9184c78f8043
COPY --from=tempo /overlay /usr/local/lib/python3.12/dist-packages
COPY bases/s2st-tencent/probe_runtime.py /opt/s2st-probe-runtime.py
RUN python3 /opt/s2st-probe-runtime.py
LABEL org.opencontainers.image.title="audio-s2st-tencent-deps"
