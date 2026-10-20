"""Fetch and verify the existing public HoloMotion V2 checkpoint bundle."""

import json
import os
import re
import sys
import time
from http.client import IncompleteRead
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .runtime.model_contract import sha256

MANIFEST_PATH = Path(__file__).with_name("frozen-model.json")


def frozen_manifest():
    return json.loads(MANIFEST_PATH.read_text())


def verify_bundle(directory):
    root = Path(directory).expanduser().resolve(strict=True)
    manifest = frozen_manifest()
    for name, expected in manifest["files"].items():
        path = root / name
        if not path.is_file() or not path.resolve().is_relative_to(root):
            raise FileNotFoundError(
                f"Frozen V2 bundle is missing {name}; use the download command"
            )
        if (
            path.stat().st_size != expected["bytes"]
            or sha256(path) != expected["sha256"]
        ):
            raise ValueError(
                f"Frozen V2 bundle size or SHA256 mismatch: {name}"
            )
    return root


def _download_file(path, url, expected, *, retries):
    partial = path.with_name(f".{path.name}.{expected['sha256'][:12]}.partial")
    invalid = partial.with_name(partial.name + ".invalid")
    last_error = None
    for attempt in range(retries + 1):
        offset = partial.stat().st_size if partial.is_file() else 0
        if offset >= expected["bytes"]:
            actual = sha256(partial)
            if offset == expected["bytes"] and actual == expected["sha256"]:
                partial.replace(path)
                return
            partial.replace(invalid)
            print(
                f"Invalid cached download kept at {invalid}: size={offset}, "
                f"SHA256={actual}; expected size={expected['bytes']}, "
                f"SHA256={expected['sha256']}",
                file=sys.stderr,
                flush=True,
            )
            offset = 0
        headers = {"Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        print(
            f"Download {path.name}: {offset}/{expected['bytes']} bytes; "
            f"attempt {attempt + 1}/{retries + 1}",
            file=sys.stderr,
            flush=True,
        )
        try:
            with urlopen(
                Request(url, headers=headers), timeout=60
            ) as response:
                append = response.status == 206
                if append:
                    content_range = response.headers.get("Content-Range", "")
                    match = re.fullmatch(
                        r"bytes (\d+)-(\d+)/(\d+)", content_range
                    )
                    if (
                        not match
                        or int(match[1]) != offset
                        or int(match[3]) != expected["bytes"]
                        or not offset <= int(match[2]) < expected["bytes"]
                    ):
                        raise ValueError(
                            f"Invalid resume Content-Range: {content_range!r}"
                        )
                elif response.status != 200:
                    raise ValueError(
                        f"Unexpected download HTTP status: {response.status}"
                    )
                elif offset:
                    print(
                        "Server ignored Range; restarting this file",
                        file=sys.stderr,
                        flush=True,
                    )
                    offset = 0
                last_progress = time.monotonic()
                with partial.open("ab" if append else "wb") as stream:
                    try:
                        while True:
                            try:
                                block = response.read(8 * 1024 * 1024)
                            except IncompleteRead as error:
                                if error.partial:
                                    stream.write(error.partial)
                                raise
                            if not block:
                                break
                            if stream.tell() + len(block) > expected["bytes"]:
                                raise ValueError(
                                    f"Download exceeds pinned size {expected['bytes']}: {path.name}"
                                )
                            stream.write(block)
                            if time.monotonic() - last_progress >= 5:
                                print(
                                    f"Download {path.name}: {stream.tell()}/{expected['bytes']} bytes",
                                    file=sys.stderr,
                                    flush=True,
                                )
                                last_progress = time.monotonic()
                    finally:
                        stream.flush()
                        os.fsync(stream.fileno())
            actual_bytes = partial.stat().st_size
            if actual_bytes != expected["bytes"]:
                raise OSError(
                    f"Incomplete download: {actual_bytes}/{expected['bytes']} bytes"
                )
            actual_hash = sha256(partial)
            if actual_hash != expected["sha256"]:
                partial.replace(invalid)
                raise ValueError(
                    f"Downloaded SHA256 mismatch for {path.name}: {actual_hash}; "
                    f"expected {expected['sha256']}. Invalid file kept at {invalid}"
                )
            partial.replace(path)
            print(
                f"Verified {path.name}: {actual_bytes} bytes",
                file=sys.stderr,
                flush=True,
            )
            return
        except (OSError, URLError, IncompleteRead, ValueError) as error:
            last_error = error
            permanent_http = (
                isinstance(error, HTTPError)
                and error.code not in (408, 429)
                and error.code < 500
            )
            if (
                attempt == retries
                or permanent_http
                or isinstance(error, ValueError)
            ):
                break
            print(
                f"Download interrupted: {error}. Retrying from the retained partial file",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(min(2**attempt, 8))
    raise RuntimeError(
        f"Download failed for {path.name}: {last_error}. "
        f"Partial file, when present, is retained at {partial}; rerun the same command to resume. "
        "HTTP 403 indicates denied access; check proxy and redirected download-host access."
    ) from last_error


def download_bundle(directory, *, retries=2):
    if type(retries) is not int or retries < 0:
        raise ValueError("Download retries must be a nonnegative integer")
    manifest = frozen_manifest()
    root = Path(directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    for name, expected in manifest["files"].items():
        path = root / name
        if (
            path.is_file()
            and path.stat().st_size == expected["bytes"]
            and sha256(path) == expected["sha256"]
        ):
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        url = (
            f"https://huggingface.co/{manifest['repository']}/resolve/"
            f"{manifest['revision']}/{manifest['subdirectory']}/{name}?download=true"
        )
        _download_file(path, url, expected, retries=retries)
    return verify_bundle(root)
