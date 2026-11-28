"""Download versioned preprocessing assets without installing external tools."""

import argparse
import hashlib
import os
import tempfile
from pathlib import Path
from urllib.request import urlopen

ASSETS = {
    "synthstrip": (
        "synthstrip.1.pt",
        "https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/requirements/synthstrip.1.pt",
        "37417f802196186441aae3e7f385d94f8a98c64a88acaeaa2723af995c653e33",
    ),
    "template": (
        "tpl-MNI152NLin2009aSym_res-1_T1w.nii.gz",
        "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin2009aSym/tpl-MNI152NLin2009aSym_res-1_T1w.nii.gz",
        "e0bdd27231960b3e930e86cf72b0d6bcf4a7d9e5195fb97e4c8f826f8d59c6e7",
    ),
    "template_mask": (
        "tpl-MNI152NLin2009aSym_res-1_desc-brain_mask.nii.gz",
        "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin2009aSym/tpl-MNI152NLin2009aSym_res-1_desc-brain_mask.nii.gz",
        "6bae185e10e6bcd871e0caedfaa88a362eaefaccab2e81b96f7fa8f36b7ad6f0",
    ),
}


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def cache_root(directory=None):
    if directory:
        return Path(directory).expanduser().resolve()
    return (
        Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "strokefill3d"
    )


def get_asset(name, directory=None):
    filename, url, expected = ASSETS[name]
    root = cache_root(directory)
    root.mkdir(parents=True, exist_ok=True)
    path = root / filename
    if path.exists():
        if sha256(path) != expected:
            raise ValueError(
                f"Cached asset checksum failed: {path}. Remove it and retry."
            )
        return path
    print(f"Downloading {name} from {url} to {path}", flush=True)
    # Separate temporary files make concurrent first-use downloads safe. Nothing
    # becomes visible at the final path until the complete bytes are verified.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=root, delete=False) as stream:
            temporary = Path(stream.name)
            with urlopen(url, timeout=120) as response:
                for block in iter(lambda: response.read(1024**2), b""):
                    stream.write(block)
        if sha256(temporary) != expected:
            raise ValueError(f"Downloaded asset checksum failed: {url}")
        os.replace(temporary, path)
    except OSError as error:
        raise RuntimeError(
            f"Cannot obtain {name}. Download {url} on a connected computer and "
            f"place it at {path}, or supply the corresponding file option."
        ) from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir")
    args = parser.parse_args()
    for name in ASSETS:
        print(get_asset(name, args.cache_dir))


if __name__ == "__main__":
    main()
