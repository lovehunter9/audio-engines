"""Install compiled vLLM dependencies beside the unchanged author Python modules."""
from importlib.metadata import distribution
from pathlib import Path
import hashlib
import json

package = distribution("vllm")
assert package.version == "0.11.1", package.version
source = Path(package.locate_file("vllm"))
target = Path("/opt/s3/vendor/vllm")
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
}, indent=2) + "\n")
print("Linked", len(records), "compiled dependencies; author Python source is unchanged")
