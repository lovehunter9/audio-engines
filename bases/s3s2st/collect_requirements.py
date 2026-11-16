"""Install the native engine requirements without alternate deployment backends."""
from importlib.metadata import requires
from pathlib import Path
from packaging.requirements import Requirement

excluded = {"torch", "torchaudio", "torchvision", "transformers", "ray", "xformers"}
selected = []
for raw in requires("vllm"):
    item = Requirement(raw)
    if item.name.lower().replace("_", "-") in excluded:
        continue
    if item.marker and not item.marker.evaluate({"extra": ""}):
        continue
    selected.append(raw)
selected.extend(["numpy==1.26.4", "opencv-python-headless==4.11.0.86"])
Path("/tmp/s3-runtime-requirements.txt").write_text("\n".join(selected) + "\n")
print("Native multiprocess, unquantized runtime dependencies:", *selected, sep="\n")
