FROM docker.io/python:3.12-slim-bookworm AS tempo
RUN python -m pip install --no-cache-dir --no-deps --target /overlay imageio-ffmpeg==0.8.1

FROM docker.io/lovehunter9/audio-s2st-tencent:deps-7bc3825f4572@sha256:bca082f5414c0f583db29a2cc7efdcc28f06bb6edb59bb51d3bf9184c78f8043 AS native-entry
ARG TENCENT_REF=d38e66aac7bb18101f417338b795e4546a5a1a59
RUN python3 -c "import os,tarfile,urllib.request; urllib.request.urlretrieve('https://codeload.github.com/lihy715/llama.cpp.Hyv3Speech/tar.gz/'+os.environ['TENCENT_REF'], '/tmp/source.tar.gz'); tarfile.open('/tmp/source.tar.gz').extractall('/src', filter='data')"
COPY bases/s2st-tencent/apply-native.py bases/s2st-tencent/native-io.patch bases/s2st-tencent/native-entry.cpp /patch/
RUN set -eu; src=$(find /src -mindepth 1 -maxdepth 1 -type d); python3 /patch/apply-native.py "$src" --shared-entry; cp /patch/native-entry.cpp "$src/entry.cpp"; g++ -std=c++17 -O3 -DNDEBUG -shared -fPIC -pthread -I"$src/common" -I"$src/include" -I"$src/ggml/include" -I"$src/tools/mtmd" -I"$src/vendor" "$src/entry.cpp" -L/opt/llama/lib -Wl,-rpath,/opt/llama/lib -lllama-common -lmtmd -lllama -lggml -o /libaudio-s2st-native.so; strip --strip-unneeded /libaudio-s2st-native.so

# This immutable multi-arch parent has the verified official native binaries and MING stack.
# vendor.Dockerfile retains the complete source bootstrap recipe.
FROM docker.io/lovehunter9/audio-s2st-tencent:deps-7bc3825f4572@sha256:bca082f5414c0f583db29a2cc7efdcc28f06bb6edb59bb51d3bf9184c78f8043
COPY --from=native-entry /libaudio-s2st-native.so /opt/llama/lib/
COPY --from=tempo /overlay /usr/local/lib/python3.12/dist-packages
COPY bases/s2st-tencent/probe_runtime.py /opt/s2st-probe-runtime.py
RUN python3 /opt/s2st-probe-runtime.py
LABEL org.opencontainers.image.title="audio-s2st-tencent-deps"
