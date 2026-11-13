FROM docker.io/pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime AS native
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential libsndfile1-dev libtbb-dev && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir "cmake>=3.21,<4" ninja "setuptools<81" wheel packaging "tbb==2021.13.1" "tbb-devel==2021.13.1"
ARG FAIRSEQ2_REVISION=c664a42ac30c5ba3ab23382327903cf690ea1ffb
RUN git clone https://github.com/facebookresearch/fairseq2.git /opt/fairseq2 && git -C /opt/fairseq2 checkout "$FAIRSEQ2_REVISION" && git -C /opt/fairseq2 submodule update --init --recursive && sed -i 's/THPVariable_Wrap(at::TensorBase var)/THPVariable_Wrap(const at::TensorBase\& var)/' /opt/fairseq2/fairseq2n/python/src/fairseq2n/bindings/type_casters/torch.cc
RUN cd /opt/fairseq2/fairseq2n && env -u CONDA_PREFIX cmake -GNinja -B build -DIconv_IS_BUILT_IN=FALSE -DIconv_INCLUDE_DIR=/opt/conda/include -DIconv_LIBRARY=/opt/conda/lib/libiconv.so -DFAIRSEQ2N_USE_CUDA=OFF -DFAIRSEQ2N_PYTHON_DEVEL=OFF -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel 2
RUN cd /opt/fairseq2/fairseq2n/python && python3 -m pip wheel --no-deps --no-build-isolation -w /wheels . && cd /opt/fairseq2 && python3 -m pip wheel --no-deps --no-build-isolation -w /wheels .

FROM docker.io/pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 libtbb12 git build-essential && rm -rf /var/lib/apt/lists/*
COPY --from=native /wheels /wheels
RUN python3 -m pip install --no-cache-dir torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128 && python3 -m pip install --no-cache-dir /wheels/*.whl "numpy<2" "tbb==2021.13.1" && rm -rf /wheels
ARG SEAMLESS_REVISION=9a081e935c29c6b1e0e05bcbf3fcb623ad306524
RUN git clone https://github.com/facebookresearch/seamless_communication.git /opt/seamless && git -C /opt/seamless checkout "$SEAMLESS_REVISION" && python3 -m pip install --no-cache-dir --no-deps /opt/seamless
RUN python3 -m pip install --no-cache-dir "datasets==2.18.0" "pyarrow==17.0.0" "simuleval==1.1.4" fire "librosa==0.11.0" openai-whisper "sonar-space==0.2.1" scipy soundfile huggingface_hub safetensors einops fastapi uvicorn httpx python-multipart websockets && apt-get purge -y build-essential gcc g++ && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*
RUN python3 -c "import torch, fairseq2, fairseq2n; from seamless_communication.streaming.agents.seamless_streaming_s2st import SeamlessStreamingS2STAgent; assert torch.__version__.startswith('2.8.'); print(SeamlessStreamingS2STAgent.__name__)" && ln -sf "$(command -v python3)" /usr/local/bin/audio-python
ENV SEAMLESS_ROOT=/opt/seamless
ENV LD_LIBRARY_PATH=/opt/conda/lib:${LD_LIBRARY_PATH}

ARG SILERO_REVISION=5cd7945676eb32225748052e2e6a0580e4686a08
RUN python3 -c "import hashlib,urllib.request; from pathlib import Path; root='https://raw.githubusercontent.com/snakers4/silero-vad/${SILERO_REVISION}/'; data=urllib.request.urlopen(root+'src/silero_vad/data/silero_vad.jit', timeout=60).read(); assert hashlib.sha256(data).hexdigest()=='e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720'; Path('/opt/seamless/silero_vad.jit').write_bytes(data); Path('/opt/seamless/silero-vad-LICENSE').write_bytes(urllib.request.urlopen(root+'LICENSE', timeout=60).read())"
