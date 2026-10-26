#!/bin/bash

set -e

CAMERA_DIR=/home/rp2/picamera2-examples/examples/imx500
cd "$CAMERA_DIR"

CAMERAS=$(rpicam-hello --list-cameras 2>&1)

if echo "$CAMERAS" | grep -qi 'imx500'; then
    echo "CAMERA: IMX500 detected"
    exec /usr/bin/python3 wildlife_camera.py \
        --model /usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk

elif echo "$CAMERAS" | grep -qi 'imx708'; then
    echo "CAMERA: Module 3 detected"
    exec /usr/bin/python3 wildlife_camera_module3.py

else
    echo "ERROR: No supported camera detected"
    echo "$CAMERAS"
    exit 1
fi
