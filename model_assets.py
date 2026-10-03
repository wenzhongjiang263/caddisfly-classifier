"""Download and verify the fixed model assets used by the cloud demo."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from urllib.parse import urlparse
from urllib.request import Request, urlopen


_DOWNLOAD_LOCK = threading.Lock()


def asset_matches(path: Path, descriptor: dict) -> bool:
    if not path.is_file() or path.stat().st_size != descriptor["bytes"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == descriptor["sha256"]


def download_asset(url: str, destination: Path, descriptor: dict) -> None:
    """Install a verified download atomically, preserving any existing file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        request = Request(url, headers={"User-Agent": "Caddisfly-Cloud-Demo"})
        with urlopen(request, timeout=30) as response, tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=".model-", suffix=".tmp", delete=False
        ) as output:
            temporary_path = Path(output.name)
            deadline = time.monotonic() + 180
            total_bytes = 0
            while True:
                if time.monotonic() > deadline:
                    raise TimeoutError("Model download exceeded three minutes.")
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > descriptor["bytes"]:
                    raise ValueError("Model download exceeds its expected size.")
                output.write(chunk)
        if not asset_matches(temporary_path, descriptor):
            raise ValueError(f"Model download failed size or SHA-256 verification: {destination.name}")
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def ensure_model_assets(root: Path, release_base_url: str) -> None:
    """Read the release manifest and fetch missing or corrupt assets once."""
    with _DOWNLOAD_LOCK:
        manifest = json.loads((root / "deployment/model_assets.json").read_text(encoding="utf-8"))
        for descriptor in manifest["assets"]:
            destination = (root / descriptor["path"]).resolve()
            if not destination.is_relative_to(root.resolve()):
                raise ValueError("Model asset path must stay inside the application.")
            if asset_matches(destination, descriptor):
                continue
            parsed = urlparse(release_base_url)
            if parsed.scheme != "https" or parsed.hostname != "github.com" or "/releases/download/" not in parsed.path:
                raise ValueError("Set deployment.release_base_url to the HTTPS download URL of your public GitHub Release.")
            filename = destination.name
            download_asset(f"{release_base_url.rstrip('/')}/{filename}", destination, descriptor)
