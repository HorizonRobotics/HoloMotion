#!/bin/bash

# example：
# ./vision_mocap_pipeline.sh example_video.mov [--static]

VIDEO="$1"
STATIC="$2"

if [[ -z "$VIDEO" ]]; then
    echo "Usage: $0 <video_path> [--static]"
    exit 1
fi

# echo "Step 1: Run Masked Droid SLAM (detect+track humans)"
# if [[ "$STATIC" == "--static" ]]; then
#     python ../../thirdparties/tram/scripts/estimate_camera.py --video "$VIDEO" --static_camera
# else
#     python ../../thirdparties/tram/scripts/estimate_camera.py --video "$VIDEO"
# fi

# echo "Step 2: Run 4D human capture with VIMO"
# python ../../thirdparties/tram/scripts/estimate_humans.py --video "$VIDEO"

echo "Step 3: Render output video"
PYTHONPATH=../../thirdparties/tram python ../../holomotion/src/data_curation/vison_mocap/visualize_tram.py --video "$VIDEO"

echo "All steps completed."