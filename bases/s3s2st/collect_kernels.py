"""Install compiled vLLM dependencies beside the unchanged author Python modules."""
from importlib.metadata import distribution
from pathlib import Path
import hashlib
import json
import shutil

package = distribution("vllm")
assert package.version.split("+", 1)[0] == "0.11.1", package.version
source = Path(package.locate_file("vllm"))
target = Path("/opt/s3/vendor/vllm")
support = []
for original in sorted((source / "vllm_flash_attn").rglob("*.py")):
    destination = target / original.relative_to(source)
    if destination.exists():
        raise FileExistsError("The author already supplies this compiled dependency entry: " + str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(original, destination)
    support.append({"path": str(original.relative_to(source)),
                    "sha256": hashlib.sha256(original.read_bytes()).hexdigest()})
if not support:
    raise RuntimeError("FlashAttention Python entry files are required beside its compiled kernels")
records = []
for library in sorted(source.rglob("*.so")):
    relative = library.relative_to(source)
    destination = target / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    destination.symlink_to(library)
    with library.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    records.append({"path": str(relative), "sha256": digest})
if not records:
    raise RuntimeError("The installed vLLM distribution contains no compiled kernels")
Path("/opt/s3/KERNELS.json").write_text(json.dumps({
    "distribution": package.version,
    "reference_build_commit": "6ac0cd370316e7c940d237e803b0eb8eda282f01",
    "exact_reference_build": False,
    "libraries": records,
    "support_files": support,
}, indent=2) + "\n")
print("Linked", len(records), "compiled dependencies; author Python source is unchanged")
