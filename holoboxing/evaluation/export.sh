#!/usr/bin/env bash
# Produce a local offline delivery. No image is pulled, tagged, or published.
set -euo pipefail
image= output=
while (($#)); do
  case "$1" in
    --image|--output-dir)
      (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
      if [[ "$1" == --image ]]; then image=$2; else output=$2; fi
      shift 2 ;;
    -h|--help)
      echo "Usage: $0 --image IMAGE --output-dir NEW_DIRECTORY"
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$image" && -n "$output" ]] || { echo "--image and --output-dir are required" >&2; exit 2; }
identity=$(docker image inspect --format '{{.Id}}' "$image")
mkdir -- "$output"
output=$(cd -- "$output" && pwd)
archive=holoboxing-eval-v11-amd64.tar.gz
docker image inspect "$image" > "$output/image-inspect.json"
printf '%s\n' "$image" > "$output/IMAGE_REFERENCE"
printf '%s\n' "$identity" > "$output/IMAGE_ID"
docker image save "$image" | gzip -1 > "$output/$archive.partial"
[[ "$(docker image inspect --format '{{.Id}}' "$image")" == "$identity" ]] || {
  echo "Image reference changed during export; delivery is incomplete" >&2; exit 1;
}
mv -- "$output/$archive.partial" "$output/$archive"
(cd -- "$output" && sha256sum "$archive" IMAGE_REFERENCE IMAGE_ID image-inspect.json > SHA256SUMS)
printf 'Offline delivery: %s\nImage ID: %s\n' "$output" "$identity"
