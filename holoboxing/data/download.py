"""Download public upstream data with resumable transfers and source receipts."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import sys
import time

from .common import decode_annotation, inside, sha256, write_json


ROBOSTRIKER_PROFILE = Path(__file__).parent / "profiles" / "robostriker.json"
INTERHUMAN_PROFILE = Path(__file__).parent / "profiles" / "interhuman.json"
INTERHUMAN_SAMPLE_LIST = (
    Path(__file__).parent / "profiles" / "interhuman_training_files.json"
)
INTERHUMAN_URL = (
    "https://drive.google.com/drive/folders/1oyozJ4E7Sqgsr7Q747Na35tWo5CjNYk3"
)


def _gdown():
    try:
        import gdown
    except ImportError as error:
        raise RuntimeError(
            "Install download dependencies: python -m pip install -r holoboxing/data/requirements.txt"
        ) from error
    version = tuple(int(v) for v in gdown.__version__.split(".")[:3])
    if version < (6, 4, 2):
        raise RuntimeError(
            "gdown>=6.4.2 is required for complete Google Drive folder discovery"
        )
    return gdown


def _md5(path):
    digest = hashlib.md5(usedforsecurity=False)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _google_download(record, partial, retries):
    import requests

    for attempt in range(retries + 1):
        offset = partial.stat().st_size if partial.is_file() else 0
        headers = {"Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            with requests.get(
                "https://drive.google.com/uc",
                params={"export": "download", "id": record["google_drive_id"]},
                headers=headers,
                stream=True,
                timeout=(10, 60),
            ) as response:
                response.raise_for_status()
                if "text/html" in response.headers.get("Content-Type", ""):
                    # Large public files can require Drive's confirmation page.
                    response.close()
                    _gdown().download(
                        id=record["google_drive_id"],
                        output=str(partial),
                        quiet=True,
                        use_cookies=False,
                        resume=True,
                        timeout=(10, 60),
                        retries=retries,
                    )
                    return
                append = response.status_code == 206 and offset > 0
                if response.status_code == 206:
                    content_range = response.headers.get("Content-Range", "")
                    if not re.fullmatch(
                        rf"bytes {offset}-\d+/\d+", content_range
                    ):
                        raise ValueError(
                            "Drive returned an invalid resume byte range"
                        )
                with partial.open("ab" if append else "wb") as stream:
                    for block in response.iter_content(256 * 1024):
                        if block:
                            stream.write(block)
                return
        except (
            requests.ConnectionError,
            requests.Timeout,
            requests.HTTPError,
        ):
            if attempt == retries:
                raise
            time.sleep(min(2**attempt, 8))


def _fetch(record, output, *, repair=False, retries=2, fetch=None):
    path = inside(output, record["path"])
    expected = record.get("sha256")
    if path.is_file():
        actual = sha256(path)
        if (
            (expected is None or actual == expected)
            and ("md5" not in record or _md5(path) == record["md5"])
            and (
                "bytes" not in record or path.stat().st_size == record["bytes"]
            )
        ):
            return {**record, "sha256": actual, "bytes": path.stat().st_size}
        if not repair:
            raise ValueError(
                f"Cached source changed: {record['path']}; use --repair to redownload"
            )
    if not re.fullmatch(r"[\w-]+", record["google_drive_id"]):
        raise ValueError("Invalid public Google Drive file ID")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    if repair:
        partial.unlink(missing_ok=True)
    if fetch is None:
        _google_download(record, partial, retries)
    else:
        fetch(record, partial)
    actual = sha256(partial)
    if expected is not None and actual != expected:
        raise ValueError(f"Downloaded source SHA256 differs: {record['path']}")
    if "md5" in record and _md5(partial) != record["md5"]:
        raise ValueError(
            f"Downloaded source differs from the public Drive checksum: {record['path']}"
        )
    if path.suffix == ".txt":
        decode_annotation(partial.read_bytes())
    if "bytes" in record and partial.stat().st_size != record["bytes"]:
        raise ValueError(f"Downloaded source size differs: {record['path']}")
    partial.replace(path)
    return {**record, "sha256": actual, "bytes": path.stat().st_size}


def _fetch_many(
    records, output, *, workers, repair, retries, quiet, fetch=None
):
    if workers < 1 or retries < 0:
        raise ValueError("Workers must be positive and retries nonnegative")
    if len({r["path"] for r in records}) != len(records):
        raise ValueError("Duplicate download paths")
    results, failures = [], []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {
            executor.submit(
                _fetch,
                record,
                output,
                repair=repair,
                retries=retries,
                fetch=fetch,
            ): record
            for record in records
        }
        for future in as_completed(pending):
            record = pending[future]
            try:
                results.append(future.result())
            except Exception as error:
                failures.append({"path": record["path"], "error": str(error)})
            completed = len(results) + len(failures)
            if not quiet and (
                completed % 25 == 0 or completed == len(records)
            ):
                print(
                    f"Download: {completed}/{len(records)}; failed: {len(failures)}",
                    file=sys.stderr,
                )
    if failures:
        write_json(
            Path(output) / "download-failed.json", {"failures": failures}
        )
        raise RuntimeError(
            f"{len(failures)} downloads failed; see download-failed.json and rerun to resume"
        )
    return sorted(results, key=lambda record: record["path"])


def download_robostriker(
    output,
    *,
    workers=4,
    limit_groups=None,
    repair=False,
    retries=2,
    quiet=False,
    fetch=None,
):
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    profile = json.loads(ROBOSTRIKER_PROFILE.read_text())
    records = profile["files"]
    if limit_groups is not None:
        if limit_groups < 1:
            raise ValueError("The source-group limit must be positive")
        groups = sorted(
            {Path(r["path"]).stem.removesuffix("_mirror") for r in records}
        )[:limit_groups]
        records = [
            r
            for r in records
            if Path(r["path"]).stem.removesuffix("_mirror") in groups
        ]
    results = _fetch_many(
        records,
        output,
        workers=workers,
        repair=repair,
        retries=retries,
        quiet=quiet,
        fetch=fetch,
    )
    report = {
        "format": "holoboxing_download_v1",
        "dataset": "robostriker",
        "source_url": profile["folder_url"],
        "upstream_repository": profile["upstream_repository"],
        "upstream_revision": profile["upstream_revision"],
        "profile_sha256": sha256(ROBOSTRIKER_PROFILE),
        "complete_source": len(results) == len(profile["files"]),
        "files": results,
    }
    write_json(output / "download.json", report)
    (output / "download-failed.json").unlink(missing_ok=True)
    return report


def normalize_drive_files(files):
    """Preserve same-name Drive copies as name(1), rather than overwriting."""
    reserved = {record["path"] for record in files}
    used, result = set(), []
    for record in files:
        original = record["path"]
        relative = Path(original)
        path = original
        number = 1
        while path in used:
            path = relative.with_name(
                f"{relative.stem}({number}){relative.suffix}"
            ).as_posix()
            number += 1
            if path in reserved:
                path = original
        used.add(path)
        result.append(
            {
                **record,
                "path": path,
                "source_path": record.get("source_path", original),
            }
        )
    return result


def discover_interhuman(output):
    output = Path(output).expanduser().resolve()
    cache = output / "source-listing.json"
    profile_hash = sha256(INTERHUMAN_PROFILE)
    if cache.is_file():
        result = json.loads(cache.read_text())
        if result.get("source_url") != INTERHUMAN_URL:
            raise ValueError("Source listing belongs to a different dataset")
        if result.get("profile_sha256") == profile_hash:
            return result["files"]
    profile = json.loads(INTERHUMAN_PROFILE.read_text())
    result = profile["files"]
    write_json(
        cache,
        {
            "source_url": INTERHUMAN_URL,
            "profile_sha256": profile_hash,
            "files": result,
        },
    )
    return result


def sample_clip_ids():
    """Load the bundled sample file list without annotation-based selection."""
    packet = json.loads(INTERHUMAN_SAMPLE_LIST.read_text())
    files = packet.get("files") if isinstance(packet, dict) else None
    if (
        not isinstance(files, list)
        or not files
        or any(
            not isinstance(name, str)
            or re.fullmatch(r"motions/[0-9]+\.pkl", name) is None
            for name in files
        )
    ):
        raise ValueError(
            "Sample files must be a nonempty list of motions/<ID>.pkl"
        )
    ids = [Path(name).stem for name in files]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate InterHuman sample clip IDs")
    return ids


def download_interhuman(
    output,
    *,
    workers=4,
    limit_clips=None,
    clip_ids=None,
    repair=False,
    retries=2,
    quiet=False,
):
    """Fetch the requested sample clips without annotation-based selection."""
    use_sample_list = clip_ids is None
    if use_sample_list:
        clip_ids = sample_clip_ids()
    if not clip_ids:
        raise ValueError("InterHuman clip IDs must not be empty")
    if limit_clips is not None and limit_clips < 1:
        raise ValueError("Clip limit must be positive")
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    listing = discover_interhuman(output)
    files = {record["path"]: record for record in listing}
    motions = [
        r
        for r in listing
        if r["path"].startswith("motions/") and r["path"].endswith(".pkl")
    ]
    if not motions:
        raise ValueError(
            "The public source contains no InterHuman motions/*.pkl"
        )
    motion_names = {Path(record["path"]).name for record in motions}
    requested = list(dict.fromkeys(clip_ids))
    available = {Path(r["path"]).stem: r for r in motions}
    if not set(requested).issubset(available):
        raise ValueError(
            f"Unknown InterHuman clip IDs: {sorted(set(requested) - available.keys())}"
        )
    motions = [available[clip] for clip in requested]
    if limit_clips is not None:
        motions = motions[:limit_clips]
    auxiliary = [
        r
        for r in listing
        if r["path"].startswith("split/") or r["path"] == "LICENSE.md"
    ]
    annotations, missing = [], []
    for record in motions:
        name = "annots/" + Path(record["path"]).stem + ".txt"
        if name in files:
            annotations.append(files[name])
        else:
            missing.append(record["path"])
    previous = output / "download.json"
    if previous.is_file():
        receipt = json.loads(previous.read_text())
        if receipt.get("dataset") != "interhuman":
            raise ValueError(
                "Download directory belongs to a different dataset"
            )
        previous_hashes = {r["path"]: r for r in receipt["files"]}

        def with_previous_hash(record):
            previous = previous_hashes.get(record["path"], {})
            if previous.get("google_drive_id") == record["google_drive_id"]:
                return {**record, "sha256": previous["sha256"]}
            return record

        annotations = [with_previous_hash(r) for r in annotations]
        auxiliary = [with_previous_hash(r) for r in auxiliary]
    downloaded = _fetch_many(
        annotations + auxiliary,
        output,
        workers=workers,
        repair=repair,
        retries=retries,
        quiet=quiet,
    )
    selected = motions
    if previous.is_file():
        selected = [with_previous_hash(r) for r in selected]
    downloaded += _fetch_many(
        selected,
        output,
        workers=workers,
        repair=repair,
        retries=retries,
        quiet=quiet,
    )
    report = {
        "format": "holoboxing_download_v1",
        "dataset": "interhuman",
        "source_url": INTERHUMAN_URL,
        "source_listing_sha256": sha256(output / "source-listing.json"),
        "source_motion_files": len(motion_names),
        "selection_mode": "clip_ids",
        "requested_clip_ids": requested,
        "sample_clip_list_sha256": sha256(INTERHUMAN_SAMPLE_LIST)
        if use_sample_list
        else None,
        "bounded_selection": limit_clips is not None,
        "selected_motion_files": len(selected),
        "missing_annotations": missing,
        "files": sorted(downloaded, key=lambda r: r["path"]),
    }
    write_json(output / "download.json", report)
    (output / "download-failed.json").unlink(missing_ok=True)
    return report
