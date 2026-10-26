# Raspberry Pi Wildlife Camera

Wildlife monitoring system supporting two Raspberry Pi cameras.

## Supported cameras

- Sony IMX500 AI Camera: on-device wildlife detection
- Camera Module 3 (IMX708): motion-triggered recording

## Features

- Automatic camera detection at startup
- Event-triggered H.264 video recording
- MP4 conversion using FFmpeg
- JPEG snapshots and event metadata
- Camera status reporting
- Google Drive archiving via rclone
- systemd service for automatic startup

## Camera switching

1. Shut down the Raspberry Pi.
2. Replace the camera module.
3. Restart the Raspberry Pi.

The startup script automatically selects the appropriate camera program.

## Files

- wildlife_camera.py: IMX500 implementation
- wildlife_camera_module3.py: Camera Module 3 implementation
- start_wildlife_camera.sh: automatic camera selection
- systemd/: service configuration
