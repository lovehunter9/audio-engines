#!/bin/sh
# Drop CUDA pieces eager inference does not load. Same RUN as `pip install torch`
# or Docker keeps the fat layer.
#
# Keep: cublas, cudnn, cuda-runtime, nvjitlink, cufft, nvrtc, curand.
# Drop: nccl (multi-GPU), cusolver/cusparse/cusparselt (solvers), cupti/nvtx
# (profiler), cufile, triton (torch.compile). Breeze's fast path already needs
# a compiler this image does not ship; eager is the contract.
set -eu

freeze=$(python3 -m pip freeze)
echo "$freeze" | grep -iE '^(nvidia-(nccl|cusolver|cusparse|cusparselt|cuda-cupti|nvtx|cufile)[^[:space:]=]*|triton|pytorch-triton)==' \
    | cut -d= -f1 \
    | while read -r pkg; do
        [ -n "$pkg" ] || continue
        python3 -m pip uninstall -y "$pkg" || true
    done || true

SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
if [ -d "$SITE" ]; then
    find "$SITE" -type d -name __pycache__ -print0 | xargs -0 -r rm -rf
    if [ -d "$SITE/nvidia" ]; then
        find "$SITE/nvidia" -type d -name include -print0 | xargs -0 -r rm -rf
        find "$SITE/nvidia" -type f \( -name '*.a' -o -name '*.h' \) -delete
    fi
    rm -rf "$SITE/torch/testing" "$SITE/torch/test" 2>/dev/null || true
fi
rm -rf /root/.cache/pip /tmp/pip-*

python3 -c "\
import torch, torchaudio
assert torch.version.cuda, 'strip_unused_cuda dropped CUDA torch'
print('stripped torch', torch.__version__, 'cuda', torch.version.cuda, 'torchaudio', torchaudio.__version__)"

python3 -c "\
import os, site
s = site.getsitepackages()[0]
print('site', s)
for name in sorted(os.listdir(s)):
    if name.startswith(('nvidia', 'torch', 'triton')):
        p = os.path.join(s, name)
        if os.path.isdir(p):
            total = 0
            for root, dirs, files in os.walk(p):
                for f in files:
                    try:
                        total += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass
            print('  %6.1f Mi  %s' % (total / 1024 / 1024, name))
"
