# audio-speakrs-ov deps (hash-tagged rebuilds). Intel only, and amd64 only: ort supports the
# OpenVINO provider on x86_64 Linux and Windows and nowhere else, so there is no second slice.
#
# This is a sibling of bases/speakrs rather than a branch inside it. That one is FROM
# nvidia/cuda, and installing Intel's compute runtime on top of the CUDA runtime image would
# carry both vendors' stacks to serve one. deps-image.sh tags a base by the hash of its own
# deps.Dockerfile, so a second base is what the mechanism is for.
#
# No torch, no NVML. speakrs runs on ONNX Runtime, and the Python shell here only serves HTTP,
# reads a file header for billing, and reports metrics. wrapper/gpu.py imports pynvml inside a
# try, so leaving nvidia-ml-py out costs a few gauges and breaks nothing -- which is why the
# import check at the bottom does not name it.
#
# Built from beclab/speakrs-diarization 267bdaa (tag pr6-267bdaa-openvino), --features openvino.
# Verified against the registry rather than the build log: this digest's
# org.opencontainers.image.revision reads 267bdaa4fdec6503e6997c45fe52cd8d88fa7d97.
#
# A digest, not a tag: it is what puts the engine's identity inside this hashed file, so the deps
# tag moves when the engine does.
ARG ENGINE_IMAGE=docker.io/beclab/speakrs-engine@sha256:f20ae45d1f88da8d977ea298da57493a20c94e0e0e1f65a995b7e99a02321967
FROM ${ENGINE_IMAGE} AS engine

FROM ubuntu:24.04
ARG DEBIAN_FRONTEND=noninteractive

# 🔴 GPU only. `ort` passes the device string through untouched and the engine accepts
# `openvino:NPU`, so the library layer supports an NPU and this image does not: the NPU plugin
# needs intel-driver-compiler-npu and intel-level-zero-npu, neither of which is installed below,
# and the device node it opens is /dev/accel, which the chart does not mount. An NPU device
# asked for here fails when the session is built, some way from the flag that asked for it.
#
# Nothing selects it today -- the chart offers cpu, intel and intel-gpu, and the compute mode
# overrides EXECUTION_MODE, so an NPU string typed into settings is discarded before it arrives.
# Written down because the two layers disagree and the disagreement is invisible from either
# one. Adding it is a base, a chart mode and a device mount, and one thing nobody has measured:
# whether the NPU plugin also refuses a static sequence length, which decides whether it wants
# the derived model or the stock one.
#
# OpenVINO's GPU plugin reaches the device through Level Zero / OpenCL, and the stock driver is
# too old to see the hardware this targets: Ubuntu 24.04 ships intel-opencl-icd 23.43, which does
# not know Arrow Lake-S (PCI 8086:7D67). The symptom is not an error -- clinfo reports 0
# platforms and the plugin quietly loads CPU only, which from the outside is indistinguishable
# from a machine with no GPU. So Intel's own compute-runtime and matching IGC are pinned here.
#
# Checksums from the upstream release notes; the .ddeb debug packages are skipped.
#
# `dpkg -i || apt-get -f` is one command, not two lines: under set -eux a bare `dpkg -i` that
# exits non-zero ends the build then and there, and the repair written on the next line could
# never run -- it read like a safety net and was dead code. It works today only because the two
# libraries these .debs depend on are installed in the apt-get above, so the first .deb that
# gains a dependency is the one that finds out. The dpkg-query loop is there because the repair
# has an outside too: `apt-get -f install` exits 0 when it decides there is nothing to fix, so a
# dpkg that failed for any other reason would leave an image with no driver in it and no failure
# anywhere in the log. The symptom then is the one this whole block exists to prevent -- 0
# platforms, CPU only, indistinguishable from a machine with no GPU.
# Recipe taken from bases/ov on the showcase/intel-openvino-asr branch, which established it
# for the ASR line on the same hardware. Said with the branch because that base is not on main:
# a reader who greps bases/ for it here finds nothing and has to decide whether the pinned
# versions below came from anywhere at all.
ARG NEO_VER=26.22.38646.4
ARG IGC_TAG=v2.36.3
ARG IGC_DEB=2.36.3+21719
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv libsndfile1 ca-certificates curl \
        ffmpeg \
        libze1 ocl-icd-libopencl1 clinfo; \
    mkdir -p /tmp/neo; cd /tmp/neo; \
    curl -fsSL -O "https://github.com/intel/intel-graphics-compiler/releases/download/${IGC_TAG}/intel-igc-core-2_${IGC_DEB}_amd64.deb"; \
    curl -fsSL -O "https://github.com/intel/intel-graphics-compiler/releases/download/${IGC_TAG}/intel-igc-opencl-2_${IGC_DEB}_amd64.deb"; \
    curl -fsSL -O "https://github.com/intel/compute-runtime/releases/download/${NEO_VER}/intel-ocloc_${NEO_VER}-0_amd64.deb"; \
    curl -fsSL -O "https://github.com/intel/compute-runtime/releases/download/${NEO_VER}/intel-opencl-icd_${NEO_VER}-0_amd64.deb"; \
    curl -fsSL -O "https://github.com/intel/compute-runtime/releases/download/${NEO_VER}/libigdgmm12_22.10.0_amd64.deb"; \
    curl -fsSL -O "https://github.com/intel/compute-runtime/releases/download/${NEO_VER}/libze-intel-gpu1_${NEO_VER}-0_amd64.deb"; \
    printf '%s\n' \
        "9e0975ac75015b431ebb2da81a802b9fd1e28a3c270313a97569cd1e6a6c6048  intel-igc-core-2_${IGC_DEB}_amd64.deb" \
        "350a52331e784bb7fb9ed42e993b5c44b7e6562fc74d2cf3102b29b6a576fa85  intel-igc-opencl-2_${IGC_DEB}_amd64.deb" \
        "25745841e66279c7ff1ff6c2d4da9ec911685c12e1c9c5609d99e4d54a364dc0  intel-ocloc_${NEO_VER}-0_amd64.deb" \
        "6fdac2e8a2aacf844ebfd90521bf7102b3ebb44f69c1bced1a9785a7ce96a3c2  intel-opencl-icd_${NEO_VER}-0_amd64.deb" \
        "6031a63d6e8a12ce61c14efc15f2c8e727061286e3820b8594e6d00615e04d54  libigdgmm12_22.10.0_amd64.deb" \
        "8bef9f24e03f826f93c076081bda13c6ac3afbd9e42b9fb8f298fab652330e2f  libze-intel-gpu1_${NEO_VER}-0_amd64.deb" \
        | sha256sum -c; \
    dpkg -i /tmp/neo/*.deb || apt-get install -y -f --no-install-recommends; \
    for p in intel-igc-core-2 intel-igc-opencl-2 intel-ocloc intel-opencl-icd \
             libigdgmm12 libze-intel-gpu1; do \
        dpkg-query -W -f='${Status}' "$p" | grep -q 'install ok installed'; \
    done; \
    rm -rf /tmp/neo /var/lib/apt/lists/*

# ffmpeg is the engine's decoder, not a convenience: speakrs takes 16 kHz mono f32 and the Rust
# side shells out to convert, so without it every diarization fails at the first request with
# "could not run ffmpeg" -- while the model loads, /v1/models answers 200 and every contract and
# capability check passes. Nothing short of sending audio finds it.

# The shell only serves HTTP and reads a file header for billing. Everything that touches audio
# samples or the model happens in the engine process.
# onnx is pinned because it is not a library this calls, it is one whose shape inference decides
# whether the derived model is accepted: the derivation runs infer_shapes and checker over a
# graph it edited, and a release that tightens either one turns batching off at startup with a
# warning nobody reads. 1.22.0 is the version the derivation case in tests/caps_smoke.py runs
# against, so the image and the check agree.
#
# onnx is here for one job: deriving the batched segmentation model OpenVINO can compile, at
# startup, from the one in the cache. Not a runtime dependency of serving -- if the import or
# the derivation fails, diar_speakrs logs it and runs segmentation one window at a time, which
# is where this backend stood before. Deriving beats baking a copy into the image: a baked one
# pins weights the engine resolves separately, and the two drift with nothing to notice.
#
# --timeout / --retries because this pulls from PyPI over whatever link the builder has, and
# pip's default 15 seconds is short enough that one slow response fails the whole image. Seen:
# a ReadTimeoutError on files.pythonhosted.org killed a build that had everything else cached.
RUN python3 -m pip install --no-cache-dir --break-system-packages --root-user-action=ignore \
        --timeout 120 --retries 10 \
        "fastapi>=0.110" "uvicorn>=0.29" python-multipart soundfile "onnx==1.22.0"

COPY --from=engine /usr/local/bin/speakrs-engine /usr/local/bin/speakrs-engine
# ONNX Runtime's OpenVINO provider and OpenVINO itself travel with the engine that dlopen's them,
# so the three can never be a version apart.
COPY --from=engine /usr/local/lib/onnxruntime/ /usr/local/lib/onnxruntime/
# ort's load-dynamic dlopen's libonnxruntime.so by an explicit search list -- the binary's own
# directory, the cwd, and cargo target dirs -- and never consults the ldconfig cache. ldconfig
# alone therefore left the engine unable to find a library sitting right there.
ENV ORT_DYLIB_PATH=/usr/local/lib/onnxruntime/libonnxruntime.so
RUN ldconfig /usr/local/lib/onnxruntime && ln -sf "$(command -v python3)" /usr/local/bin/audio-python

# Import-check the shell's deps, and prove the engine binary is the right architecture and runs.
# A wrong-arch copy survives build and push and only shows up as "exec format error" in a pod.
# ffmpeg is checked here and not left to a probe on purpose: a shell-out dependency shows up in
# no import, no requirements file and no health endpoint, so the first thing that notices is a
# request carrying audio -- long after the image shipped and the engine reported itself ready.
#
# clinfo is checked for the same reason and reports the failure this base exists to prevent: a
# driver that cannot see the card leaves it printing zero platforms, and every other check here
# still passes. It is printed rather than asserted, because the build host has no Intel GPU --
# the assertion belongs on the machine, and the engine makes it there by counting node placement.
RUN python3 -c "import fastapi, uvicorn, multipart, soundfile; \
    print('soundfile', soundfile.__libsndfile_version__)" \
    && ffmpeg -version | head -1 \
    && ffprobe -version | head -1 \
    && (clinfo -l || true) \
    && test -e /usr/local/lib/onnxruntime/libonnxruntime_providers_openvino.so \
    && test -e /usr/local/lib/onnxruntime/libopenvino_intel_gpu_plugin.so \
    && /usr/local/bin/speakrs-engine --version

LABEL org.opencontainers.image.title="audio-speakrs-ov-deps"
