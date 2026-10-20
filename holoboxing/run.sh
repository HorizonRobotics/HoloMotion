#!/usr/bin/env bash
# Student-facing entrypoint; inference and playback use the bundled image.
set -euo pipefail

usage() {
  cat <<'EOF'
HoloBoxing: validate a model, run a match, or watch a replay.

  ./holoboxing/run.sh --init
  ./holoboxing/run.sh --validate models/my_policy.onnx
  ./holoboxing/run.sh --match models/my_policy.onnx models/opponent_policy.onnx
  ./holoboxing/run.sh --view results/match/replay.hdf5

Choose exactly one task:
  --init                 Reserved setup entry; pull the image as shown in README.
  --validate MODEL       Check one ONNX model; write results/validation.json.
  --match YOUR_POLICY OPPONENT_POLICY
                         Run your model against an opponent; write results/match/.
  --view REPLAY          Watch replay.hdf5 or match.json at localhost:8080.

Options:
  --image IMAGE          Provided image tag/ID (default: HOLOBOXING_IMAGE,
                         or holoboxing-eval:local-v10). No automatic download.
  --output PATH          Validation JSON filename or new match directory.
  --gpu INDEX            Host GPU index for validation/match (default: 0).
  --qualification MODE   local or official; validation only (default: local).
  --port PORT            Local browser port; view only (default: 8080).
  --assets DIRECTORY     Optional replacement; bundled resources are default.
  --lower DIRECTORY      Optional replacement; bundled V2 controller is default.
  --print-command        Show the command without running Docker.
  -h, --help             Show this help.

Linux and Docker are required. Validation/matches also need an NVIDIA GPU
and NVIDIA Container Toolkit. Playback needs no GPU. Models, resources and
the player are already in the provided image; Python/npm setup is unnecessary.
Use your own model as YOUR_POLICY and a model to test against as OPPONENT_POLICY.
The first model controls the red robot; the second controls the blue robot.
Keep the replay browser tab visible during MP4 export. Stop the viewer with Ctrl+C.
Existing results are never overwritten. Choose another --output for a new run.
EOF
}

fail() { printf 'Error: %s\nRun %s --help for examples.\n' "$1" "$0" >&2; exit 2; }
select_task() { [[ -z "$task" ]] || fail 'Choose only one of --init, --validate, --match and --view.'; task=$1; }
file_path() { [[ -f "$1" ]] || fail "File does not exist: $1"; realpath -e -- "$1"; }
directory_path() { [[ -d "$1" ]] || fail "Directory does not exist: $1"; realpath -e -- "$1"; }

task= image=${HOLOBOXING_IMAGE:-holoboxing-eval:local-v10} output= gpu=0 port=8080
qualification=local assets= lower= print_command=0
qualification_set=0 port_set=0 gpu_set=0
red= blue= replay=
while (($#)); do
  case "$1" in
    --init) select_task init; shift ;;
    --validate) (($# >= 2)) || fail '--validate needs a model file.'; select_task validate; red=$2; shift 2 ;;
    --match) (($# >= 3)) || fail '--match needs your model and an opponent model.'; select_task match; red=$2; blue=$3; shift 3 ;;
    --view) (($# >= 2)) || fail '--view needs a replay file.'; select_task view; replay=$2; shift 2 ;;
    --image|--output|--gpu|--qualification|--port|--assets|--lower)
      (($# >= 2)) && [[ -n "$2" ]] || fail "$1 needs a value."
      case "$1" in
        --image) image=$2 ;; --output) output=$2 ;;
        --gpu) gpu=$2; gpu_set=1 ;;
        --qualification) qualification=$2; qualification_set=1 ;;
        --port) port=$2; port_set=1 ;;
        --assets) assets=$2 ;; --lower) lower=$2 ;;
      esac
      shift 2 ;;
    --print-command) print_command=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) fail "Unknown option: $1" ;;
  esac
done
[[ -n "$task" ]] || { usage; exit 2; }
if [[ "$task" == init ]]; then
  fail 'Initialization is reserved for a future release. Pull the published image with the Docker command in README. No installation or download was performed.'
fi
[[ "$gpu" =~ ^[0-9]+$ ]] || fail '--gpu must be a nonnegative GPU index.'
[[ "$port" =~ ^[0-9]+$ ]] && ((${#port} <= 5)) && ((10#$port >= 1 && 10#$port <= 65535)) || fail '--port must be between 1 and 65535.'
[[ "$qualification" == local || "$qualification" == official ]] || fail '--qualification must be local or official.'
[[ "$task" == validate || "$qualification_set" == 0 ]] || fail '--qualification is only used with --validate.'
[[ "$task" == view || "$port_set" == 0 ]] || fail '--port is only used with --view.'
if [[ "$task" == view ]]; then
  [[ -z "$output" && -z "$lower" && "$gpu_set" == 0 ]] || fail '--view does not use --output, --lower or --gpu.'
fi

command=(docker run --rm --read-only --tmpfs /tmp:rw,exec --user "$(id -u):$(id -g)")
runtime=()
if [[ -n "$assets" ]]; then
  assets=$(directory_path "$assets")
  command+=(--mount "type=bind,src=$assets,dst=/assets,readonly")
  runtime+=(--assets /assets)
fi
if [[ -n "$lower" ]]; then
  lower=$(directory_path "$lower")
  command+=(--mount "type=bind,src=$lower,dst=/lower,readonly")
  runtime+=(--lower /lower)
fi
if [[ "$task" == view ]]; then
  replay=$(file_path "$replay")
  name=$(basename -- "$replay")
  command+=(--env NVIDIA_VISIBLE_DEVICES=void -p "127.0.0.1:$port:8080"
    --mount "type=bind,src=$replay,dst=/inputs/$name,readonly"
    "$image" view "/inputs/$name" --host 0.0.0.0 "${runtime[@]}")
else
  red=$(file_path "$red")
  command+=(--network none --gpus "device=$gpu"
    --mount "type=bind,src=$red,dst=/inputs/red.onnx,readonly")
  if [[ "$task" == validate ]]; then
    output=${output:-results/validation.json}
    [[ ! -e "$output" && ! -L "$output" ]] || fail "Output already exists: $output. Choose a new --output."
    output_name=$(basename -- "$output")
    output_directory=$(realpath -m -- "$(dirname -- "$output")")
    runtime+=(--qualification "$qualification" --output "/results/$output_name")
    arguments=(validate /inputs/red.onnx)
  else
    blue=$(file_path "$blue")
    output_directory=$(realpath -m -- "${output:-results/match}")
    for name in match.json replay.hdf5; do
      [[ ! -e "$output_directory/$name" && ! -L "$output_directory/$name" ]] || fail "Output already exists: $output_directory/$name. Choose a new --output."
    done
    command+=(--mount "type=bind,src=$blue,dst=/inputs/blue.onnx,readonly")
    runtime+=(--output-dir /results)
    arguments=(match /inputs/red.onnx /inputs/blue.onnx)
  fi
  if [[ "$print_command" == 0 ]]; then mkdir -p -- "$output_directory"; fi
  command+=(--mount "type=bind,src=$output_directory,dst=/results" "$image" "${arguments[@]}" "${runtime[@]}")
fi
if [[ "$print_command" == 1 ]]; then
  printf '%q ' "${command[@]}"; printf '\n'; exit 0
fi
command -v docker >/dev/null || fail 'Docker is not installed or not on PATH.'
docker image inspect "$image" >/dev/null 2>&1 || fail "Image unavailable: $image. Load the provided image, or choose its tag with --image."
contract=$(docker image inspect --format '{{index .Config.Labels "org.holomotion.holoboxing.robot-contract"}} {{index .Config.Labels "org.holomotion.holoboxing.collision-precision"}}' "$image")
[[ "$contract" == 'urdf-gloves-v1 epa-plane-fp64-v1' ]] || fail 'Image predates the gloved robot and collision precision fixes. Rebuild from the current source.'
if [[ "$task" == view ]]; then
  printf 'Open http://localhost:%s in your browser. Stop with Ctrl+C.\n' "$port"
else
  printf 'Output directory: %s\n' "$output_directory"
fi
exec "${command[@]}"
