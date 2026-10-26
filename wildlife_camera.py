import argparse
import sys
from functools import lru_cache
import time
import cv2
from datetime import datetime
from pathlib import Path
import queue
import threading
import json
from picamera2.encoders import H264Encoder
from picamera2.outputs import CircularOutput
import subprocess
import fcntl


CAPTURE_DIR = Path.home() / "wildlife"
CAPTURE_DIR.mkdir(exist_ok=True)
VIDEO_DIR = CAPTURE_DIR / "video"
VIDEO_DIR.mkdir(exist_ok=True)
SAVE_DNG = False
SNAPSHOT_FILE = CAPTURE_DIR / "snapshot.jpg"
SNAPSHOT_INTERVAL = 1.0
ORIENTATION_FILE = CAPTURE_DIR / "orientation.json"
DETECTION_LOG = CAPTURE_DIR / "detections.csv"

WILDLIFE_THRESHOLDS = {
    "bird": 0.40,
    "cat": 0.75,
    "dog": 0.70,
}

DEFAULT_WILDLIFE_THRESHOLD = 0.60


video_lock = threading.Lock()

video_recording = False
last_detection_log = 0.0
video_orientation = 0
video_stop_time = 0
video_start_time = 0
video_filename = None

VIDEO_AFTER_TRIGGER = 25
VIDEO_MAX_DURATION = 120
VIDEO_COOLDOWN = 10

video_cooldown_until = 0

STATUS_FILE = Path("/home/rp2/wildlife/camera_status.json")

status_lock = threading.Lock()

camera_status = {
    "state": "monitoring",
    "heartbeat": None,
    "last_trigger": None,
    "last_species": None,
    "last_confidence": None,
    "event_count": 0
}

if STATUS_FILE.exists():
    try:
        previous = json.loads(STATUS_FILE.read_text())

        for key in (
            "last_trigger",
            "last_species",
            "last_confidence",
            "event_count",
        ):
            camera_status[key] = previous.get(
                key,
                camera_status[key]
            )

    except (OSError, ValueError) as exc:
        print(f"STATUS: could not restore previous state: {exc}")

def log_detections(detections):
    """Append AI detections to a CSV file."""

    import csv

    timestamp = datetime.now().isoformat(timespec="milliseconds")
    labels = get_labels()

    new_file = not DETECTION_LOG.exists()

    with open(DETECTION_LOG, "a", newline="") as f:
        writer = csv.writer(f)

        if new_file:
            writer.writerow([
                "timestamp",
                "class",
                "confidence",
                "x", "y", "width", "height"
            ])

        for detection in detections:
            x, y, w, h = detection.box

            writer.writerow([
                timestamp,
                labels[int(detection.category)],
                round(float(detection.conf), 4),
                int(x), int(y), int(w), int(h)
            ])        


def get_orientation():
    try:
        with open(ORIENTATION_FILE) as f:
            angle = int(json.load(f)["orientation"])

        return angle if angle in (0, 90, 180, 270) else 0

    except (OSError, ValueError, KeyError, TypeError):
        return 0

def rotate_image(filename, angle):
    """Rotate a saved JPEG clockwise by the specified angle."""

    if angle == 0:
        return

    rotations = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }

    frame = cv2.imread(str(filename))

    if frame is None:
        raise RuntimeError(f"Cannot read image: {filename}")

    frame = cv2.rotate(frame, rotations[angle])

    temp_file = filename.with_name(
        filename.stem + ".rotating.jpg"
    )

    if not cv2.imwrite(str(temp_file), frame):
        raise RuntimeError(f"Cannot write rotated image: {temp_file}")

    temp_file.replace(filename)        


def update_camera_status(**changes):
    with status_lock:
        camera_status.update(changes)
        camera_status["heartbeat"] = datetime.now().isoformat(
            timespec="seconds"
        )

        temp_file = STATUS_FILE.with_suffix(".tmp")

        try:
            temp_file.write_text(json.dumps(camera_status, indent=2))
            temp_file.replace(STATUS_FILE)
        except OSError as exc:
            print(f"STATUS: write failed: {exc}")

from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics, postprocess_nanodet_detection

last_detections = []

#bird_frames = 0
#last_bird_trigger = 0
#last_bird_seen = 0
BIRD_GAP = 1.0
BIRD_FRAMES_REQUIRED = 3
BIRD_COOLDOWN = 10
capture_queue = queue.Queue()
video_queue = queue.Queue()
WILDLIFE_CLASSES = {
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
}

WILDLIFE_FRAMES_REQUIRED = BIRD_FRAMES_REQUIRED
WILDLIFE_GAP = BIRD_GAP
WILDLIFE_COOLDOWN = BIRD_COOLDOWN

wildlife_frames = {
    name: 0 for name in WILDLIFE_CLASSES
}

last_wildlife_seen = {
    name: 0.0 for name in WILDLIFE_CLASSES
}

last_wildlife_trigger = {
    name: 0.0 for name in WILDLIFE_CLASSES
}


class Detection:
    def __init__(self, coords, category, conf, metadata):
        """Create a Detection object, recording the bounding box, category and confidence."""
        self.category = category
        self.conf = conf
        self.box = imx500.convert_inference_coords(coords, metadata, picam2)


def parse_detections(metadata: dict):
    """Parse the output tensor into a number of detected objects, scaled to the ISP output."""
    global last_detections
    bbox_normalization = intrinsics.bbox_normalization
    bbox_order = intrinsics.bbox_order
    threshold = args.threshold
    iou = args.iou
    max_detections = args.max_detections

    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    input_w, input_h = imx500.get_input_size()
    if np_outputs is None:
        return last_detections
    if intrinsics.postprocess == "nanodet":
        boxes, scores, classes = postprocess_nanodet_detection(
            outputs=np_outputs[0], conf=threshold, iou_thres=iou, max_out_dets=max_detections
        )[0]
        from picamera2.devices.imx500.postprocess import scale_boxes

        boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
    else:
        boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
        if bbox_normalization:
            boxes = boxes / input_h

        if bbox_order == "xy":
            boxes = boxes[:, [1, 0, 3, 2]]

    last_detections = [
        Detection(box, category, score, metadata) for box, score, category in zip(boxes, scores, classes) if score > threshold
    ]
    return last_detections


@lru_cache
def get_labels():
    labels = intrinsics.labels

    if intrinsics.ignore_dash_labels:
        labels = [label for label in labels if label and label != "-"]
    return labels

def wildlife_event(class_name, confidence, box):
    now_dt = datetime.now()
    timestamp = now_dt.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    timestamp_iso = now_dt.isoformat(timespec="milliseconds")

    print(
        f"*** WILDLIFE EVENT *** "
        f"{class_name} confidence={confidence:.2f} "
        f"time={timestamp}"
    )

    with status_lock:
        event_count = camera_status["event_count"] + 1
    
    update_camera_status(
        last_trigger=timestamp_iso,
        last_species=class_name,
        last_confidence=float(confidence),
        event_count=event_count
    )

    capture_queue.put({
        "timestamp": timestamp,
        "timestamp_iso": timestamp_iso,
        "class_name": class_name,
        "confidence": float(confidence),
        "box": [int(v) for v in box],
    })

    trigger_video(timestamp)  

def snapshot_worker():
    print("SNAPSHOT WORKER: started")

    while True:
        try:
            temp_file = SNAPSHOT_FILE.with_suffix(".tmp.jpg")

            picam2.capture_file(
                str(temp_file),
                name="main",
            )

            angle = get_orientation()

            if angle:
                frame = cv2.imread(str(temp_file))

                if frame is None:
                    raise RuntimeError("Could not read snapshot")

                rotations = {
                    90: cv2.ROTATE_90_CLOCKWISE,
                    180: cv2.ROTATE_180,
                    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
                }

                frame = cv2.rotate(frame, rotations[angle])

                if not cv2.imwrite(str(temp_file), frame):
                    raise RuntimeError("Could not save rotated snapshot")

            temp_file.replace(SNAPSHOT_FILE)

        except Exception as e:
            print(f"SNAPSHOT ERROR: {e}")

        time.sleep(SNAPSHOT_INTERVAL)

def rotate_box(box, angle, width, height):
    """Rotate an (x, y, w, h) bounding box clockwise."""

    x, y, w, h = map(int, box)

    if angle == 90:
        return [height - y - h, x, h, w]

    if angle == 180:
        return [width - x - w, height - y - h, w, h]

    if angle == 270:
        return [y, width - x - w, h, w]

    return [x, y, w, h]        
        
def capture_worker():
    while True:
        event = capture_queue.get()

        try:
            timestamp = event["timestamp"]
            class_name = event["class_name"]
            event_id = f"{timestamp}_{class_name}"

            filename = (
                CAPTURE_DIR /
                f"{timestamp}_{class_name}.jpg"
            )

            print(f"CAPTURE: {filename}")

            jpeg_filename = (
                CAPTURE_DIR /
                f"{event_id}.jpg"
            )
            
            dng_filename = (
                CAPTURE_DIR /
                f"{event_id}.dng"
            )
            
            json_filename = (
                CAPTURE_DIR /
                f"{event_id}.json"
            )

            json_filename = filename.with_suffix(".json")

            print(f"CAPTURE: {jpeg_filename}")
            
            request = picam2.capture_request()
            
            try:
                metadata = request.get_metadata()
            
                request.save(
                    "main",
                    str(jpeg_filename)
                )

                angle = get_orientation()
                
                if SAVE_DNG:
                    request.save_dng(
                        str(dng_filename)
                    )
            
            finally:
                request.release()

            rotate_image(jpeg_filename, angle)  

            rotated_box = rotate_box(
                event["box"],
                angle,
                1920,
                1080
            )
               
            jpeg_width, jpeg_height = (
                (1080, 1920)
                if angle in (90, 270)
                else (1920, 1080)
            )   

            
            event_data = {
                "event_id": event_id,
                
                "timestamp": event["timestamp"],
                "timestamp_iso": event["timestamp_iso"],
                
                "class": event["class_name"],
                "confidence": event["confidence"],
                "bounding_box": event["box"],
                
                "image": jpeg_filename.name,
                
                "raw_image": (
                    dng_filename.name
                    if SAVE_DNG
                    else None
                ),
                
                "video": f"{event_id}.mp4",
                "video_source": f"video/{event_id}.h264",
                
                "jpeg_width": jpeg_width,
                "jpeg_height": jpeg_height,
                
                "raw_width": 4056,
                "raw_height": 3040,
           
               # Camera metadata
                "exposure_time_us": metadata.get("ExposureTime"),
                "analogue_gain": metadata.get("AnalogueGain"),
                "digital_gain": metadata.get("DigitalGain"),
                "colour_temperature": metadata.get("ColourTemperature"),
                "sensor_timestamp_ns": metadata.get("SensorTimestamp"),
                "lux": metadata.get("Lux"),
                "focus_figure": metadata.get("FocusFoM"),
            }
            
            with open(json_filename, "w") as f:
                json.dump(
                    event_data,
                    f,
                    indent=2
                )

            print(f"SAVED: {jpeg_filename}")
            print(f"SAVED: {dng_filename}")
            print(f"SAVED: {json_filename}") 

        except Exception as e:
            print(f"CAPTURE ERROR: {e}")

        finally:
            capture_queue.task_done()


def status_worker():
    while True:
        with video_lock:
            recording = video_recording

        update_camera_status(
            state="recording" if recording else "monitoring"
        )

        time.sleep(5)            

def video_worker():
    global video_recording
    global video_stop_time
    global video_start_time
    global video_filename
    global video_orientation
    global video_cooldown_until

    print("VIDEO WORKER: started")

    while True:
        time.sleep(0.25)

        with video_lock:
            if (
                video_recording
                and (
                    time.time() >= video_stop_time
                    or time.time() - video_start_time >= VIDEO_MAX_DURATION
                )
            ):
                print("Stopping wildlife video...")

                circular_output.stop()

                print(
                    f"VIDEO ELAPSED: "
                    f"{time.time() - video_start_time:.1f}s"
                )

                video_recording = False
                video_cooldown_until = time.time() + VIDEO_COOLDOWN

                update_camera_status(state="monitoring")

                print("VIDEO SAVED")

                video_queue.put((
                    video_filename,
                    video_orientation
                ))

                print("VIDEO: queued for MP4 conversion")

def trigger_video(timestamp):
    global video_recording
    global video_stop_time
    global video_start_time
    global video_filename
    global video_orientation
    global video_cooldown_until
    

    

    now = time.time()

    

    with video_lock:

        # Bird seen while already recording:
        # extend the recording.

        if now < video_cooldown_until:
            return
            
        if video_recording:
            video_stop_time = min(
                now + VIDEO_AFTER_TRIGGER,
                video_start_time + VIDEO_MAX_DURATION
            )
        
            print("VIDEO: extended")
            return

        filename = (
            VIDEO_DIR /
            f"{timestamp}_bird.h264"
        )

        video_filename = filename
        video_orientation = get_orientation()
        

        circular_output.fileoutput = str(filename)
        circular_output.start()
        video_start_time = now

        video_recording = True
        update_camera_status(state="recording")
        video_stop_time = (
            now + VIDEO_AFTER_TRIGGER
        )

        print(
            f"VIDEO STARTED: {filename}"
        )                                


def make_mp4(h264_filename, angle=0):
    """Convert H.264 to MP4, applying clockwise rotation."""

    h264_filename = Path(h264_filename)
    mp4_filename = h264_filename.with_suffix(".mp4")

    # Write to a temporary MP4 first
    temp_filename = mp4_filename.with_name(
        mp4_filename.stem + ".tmp.mp4"
    )

    rotations = {
        90: "transpose=clock",
        180: "hflip,vflip",
        270: "transpose=cclock",
    }

    try:
        command = [
            "ffmpeg",
            "-y",
            "-loglevel", "error",
            "-i", str(h264_filename),
        ]

        if angle == 0:
            # No rotation: preserve original video quality
            command += [
                "-c:v", "copy",
                "-bsf:v",
                "setts=ts='N*1000000/10':time_base=1/1000000",
            ]
        else:
            # Rotation requires re-encoding
            command += [
                "-vf", rotations[angle],
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "20",
                "-pix_fmt", "yuv420p",
            ]

        command += [
            "-movflags", "+faststart",
            str(temp_filename),
        ]

        subprocess.run(
            command,
            check=True,
        )

        # Replace only after successful conversion
        temp_filename.replace(mp4_filename)

        print(
            f"MP4 SAVED: {mp4_filename} "
            f"orientation={angle}"
        )

    except (subprocess.CalledProcessError, OSError) as e:
        print(f"MP4 ERROR: {e}")

def is_archived(h264_filename):
    """Check whether a recording is already archived."""

    manifest = VIDEO_DIR.parent / "archived_mp4.txt"

    if not manifest.exists():
        return False

    mp4_name = Path(h264_filename).with_suffix(".mp4").name

    with manifest.open("r") as f:
        return mp4_name in {line.strip() for line in f}


def convert_with_lock(filename, angle):
    """Prevent conversion and archive cleanup from overlapping."""

    lock_path = VIDEO_DIR.parent / "video-conversion.lock"

    with lock_path.open("a") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)

        try:
            if is_archived(filename):
                print(f"VIDEO CONVERSION: already archived, skipping {filename}")
                return

            make_mp4(filename, angle)
            
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
        

def video_conversion_worker():
    """Convert completed H.264 recordings to MP4 in the background."""

    print("VIDEO CONVERSION WORKER: started")

    while True:
        filename, angle = video_queue.get()

        try:
            print(
                f"VIDEO CONVERSION: {filename} "
                f"orientation={angle}"
            )

            convert_with_lock(filename, angle=angle)

        except Exception as e:
            print(f"VIDEO CONVERSION ERROR: {e}")

        finally:
            video_queue.task_done()      

def recover_video_conversions():
    """Queue incomplete conversions, excluding archived recordings."""

    archived_manifest = VIDEO_DIR.parent / "archived_mp4.txt"

    archived = set()

    if archived_manifest.exists():
        with archived_manifest.open("r") as f:
            archived = {
                line.strip()
                for line in f
                if line.strip()
            }

    recovered = 0
    skipped = 0

    for h264_file in sorted(VIDEO_DIR.glob("*.h264")):

    # Ignore empty or incomplete recordings
        if h264_file.stat().st_size == 0:
            print(f"VIDEO RECOVERY: skipping empty file {h264_file.name}")
            continue

        mp4_file = h264_file.with_suffix(".mp4")

        # Do not regenerate recordings already archived
        if mp4_file.name in archived:
            skipped += 1
            continue

        # Existing completed MP4
        if mp4_file.exists() and mp4_file.stat().st_size > 0:
            continue

        # Recover incomplete conversion
        angle = get_orientation()

        video_queue.put((h264_file, angle))
        recovered += 1

        print(
            f"VIDEO RECOVERY: queued {h264_file.name} "
            f"orientation={angle}"
        )

    print(
        f"VIDEO RECOVERY: {recovered} queued, "
        f"{skipped} already archived"
    )       
        
def draw_detections(request, stream="main"):
    """Draw the detections for this request onto the ISP output."""

    global last_detection_log
        
    detections = last_results
    if detections is None:
        return
        
    now = time.monotonic()
        
    if now - last_detection_log >= 1.0:
        log_detections(detections)
        last_detection_log = now  

            
    labels = get_labels()
    with MappedArray(request, stream) as m:
        for detection in detections:
            x, y, w, h = detection.box
            class_name = labels[int(detection.category)]
            label = f"{class_name} ({detection.conf:.2f})"
            
            if class_name in WILDLIFE_CLASSES:

                required_confidence = WILDLIFE_THRESHOLDS.get(
                    class_name,
                    DEFAULT_WILDLIFE_THRESHOLD
                )

                if detection.conf >= required_confidence:

                    now = time.time()

                    # Reset confirmation after a long detection gap
                    if now - last_wildlife_seen[class_name] > WILDLIFE_GAP:
                        wildlife_frames[class_name] = 0

                    # Count detections for this animal class
                    wildlife_frames[class_name] = min(
                        wildlife_frames[class_name] + 1,
                        WILDLIFE_FRAMES_REQUIRED
                    )

                    last_wildlife_seen[class_name] = now

                    print(
                        f"WILDLIFE: {class_name} "
                        f"confidence={detection.conf:.2f} "
                        f"frames={wildlife_frames[class_name]}/"
                        f"{WILDLIFE_FRAMES_REQUIRED}"
                    )

                    # Trigger an event when confirmed and cooldown expired
                    if (
                        wildlife_frames[class_name] >= WILDLIFE_FRAMES_REQUIRED
                        and now - last_wildlife_trigger[class_name]
                            >= WILDLIFE_COOLDOWN
                    ):

                        wildlife_event(
                            class_name,
                            detection.conf,
                            detection.box
                        )

                        last_wildlife_trigger[class_name] = now

                        wildlife_frames[class_name] = 0
                    


            # Calculate text size and position
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_x = x + 5
            text_y = y + 15

            # Create a copy of the array to draw the background with opacity
            overlay = m.array.copy()

            # Draw the background rectangle on the overlay
            cv2.rectangle(
                overlay,
                (text_x, text_y - text_height),
                (text_x + text_width, text_y + baseline),
                (255, 255, 255),  # Background color (white)
                cv2.FILLED,
            )

            alpha = 0.30
            cv2.addWeighted(overlay, alpha, m.array, 1 - alpha, 0, m.array)

            # Draw text on top of the background
            cv2.putText(m.array, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Draw detection box
            cv2.rectangle(m.array, (x, y), (x + w, y + h), (0, 255, 0, 0), thickness=2)


            
        if intrinsics.preserve_aspect_ratio:
            b_x, b_y, b_w, b_h = imx500.get_roi_scaled(request)
            color = (255, 0, 0)  # red
            cv2.putText(m.array, "ROI", (b_x + 5, b_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.rectangle(m.array, (b_x, b_y), (b_x + b_w, b_y + b_h), (255, 0, 0, 0))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="Path of the model",
        default="/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk",
    )
    parser.add_argument("--fps", type=int, help="Frames per second")
    parser.add_argument("--bbox-normalization", action=argparse.BooleanOptionalAction, help="Normalize bbox")
    parser.add_argument(
        "--bbox-order", choices=["yx", "xy"], default="yx", help="Set bbox order yx -> (y0, x0, y1, x1) xy -> (x0, y0, x1, y1)"
    )
    parser.add_argument("--threshold", type=float, default=0.3, help="Detection threshold")
    parser.add_argument("--iou", type=float, default=0.65, help="Set iou threshold")
    parser.add_argument("--max-detections", type=int, default=10, help="Set max detections")
    parser.add_argument("--ignore-dash-labels", action=argparse.BooleanOptionalAction, help="Remove '-' labels ")
    parser.add_argument("--postprocess", choices=["", "nanodet"], default=None, help="Run post process of type")
    parser.add_argument(
        "-r",
        "--preserve-aspect-ratio",
        action=argparse.BooleanOptionalAction,
        help="preserve the pixel aspect ratio of the input tensor",
    )
    parser.add_argument("--labels", type=str, help="Path to the labels file")
    parser.add_argument("--print-intrinsics", action="store_true", help="Print JSON network_intrinsics then exit")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # This must be called before instantiation of Picamera2
    imx500 = IMX500(args.model)
    intrinsics = imx500.network_intrinsics
    if not intrinsics:
        intrinsics = NetworkIntrinsics()
        intrinsics.task = "object detection"
    elif intrinsics.task != "object detection":
        print("Network is not an object detection task", file=sys.stderr)
        exit()

    print(f"INFERENCE RATE: {intrinsics.inference_rate}")    

    # Override intrinsics from args
    for key, value in vars(args).items():
        if key == 'labels' and value is not None:
            with open(value, 'r') as f:
                intrinsics.labels = f.read().splitlines()
        elif hasattr(intrinsics, key) and value is not None:
            setattr(intrinsics, key, value)

    # Defaults
    if intrinsics.labels is None:
        with open("assets/coco_labels.txt", "r") as f:
            intrinsics.labels = f.read().splitlines()
    intrinsics.update_with_defaults()

    if args.print_intrinsics:
        print(intrinsics)
        exit()

    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_video_configuration(
        main={
            "size": (1920, 1080),
            "format": "XRGB8888"
        },
        lores={
            "size": (640, 480),
            "format": "YUV420"
        },
        raw={
            "size": (4056, 3040)
        },
        controls={
            "FrameRate": 10
        },
        buffer_count=12
    )

    print("PREVIEW CONFIG:")
    print(config)

    video_encoder = H264Encoder(
        bitrate=10_000_000,
        iperiod=10
    )

    
    imx500.show_network_fw_progress_bar()
    picam2.start(config, show_preview=False)

    circular_output = CircularOutput(
        buffersize=50
    )
    
    picam2.start_encoder(
        video_encoder,
        circular_output
    )

    video_thread = threading.Thread(
        target=video_worker,
        daemon=True
    )
    
    video_thread.start()

    threading.Thread(
        target=snapshot_worker,
        daemon=True,
    ).start()

    capture_thread = threading.Thread(
            target=capture_worker,
            daemon=True
        )
            
    capture_thread.start()

    threading.Thread(
        target=video_conversion_worker,
        daemon=True
    ).start()

    recover_video_conversions()

    threading.Thread(
        target=status_worker,
        daemon=True
    ).start()

    if intrinsics.preserve_aspect_ratio:
        imx500.set_auto_aspect_ratio()

    last_results = None
    picam2.pre_callback = draw_detections
    while True:
        metadata = picam2.capture_metadata()
    
        frame_duration = metadata.get(
            "FrameDuration"
        )
    
        last_results = parse_detections(
            metadata
        )
