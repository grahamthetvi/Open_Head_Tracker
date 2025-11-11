import cv2
import numpy as np
import os
import mediapipe as mp
import time
import math
from scipy.spatial.transform import Rotation as Rscipy
from collections import deque
import pyautogui
import threading
import keyboard

# Configure PyAutoGUI for head-based mouse control
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0  # Remove delay between PyAutoGUI commands
import json
from datetime import datetime
from pathlib import Path
import socket  # For UDP communication with OptiKey

"""
=============================================================================
HEAD TRACKING SYSTEM EXPLANATION
=============================================================================

HOW HEAD DIRECTION IS DETERMINED: (PRO FIX METHOD)

1. MediaPipe Face Mesh Input:
   - Provides 468 3D landmarks on the face.
   - We select 6 key landmarks (nose, chin, eye corners, mouth corners).

2. Define 3D Canonical Model:
   - We use a standard 3D model of a face that defines the (X, Y, Z)
     coordinates of those 6 landmarks in a "neutral" 3D space.

3. Estimate Camera Properties:
   - We create a virtual "camera matrix" by estimating the focal length
     and optical center based on the webcam's frame width and height.

4. SolvePnP (Perspective-n-Point):
   - We feed the 3D model points, the 2D image points, and the camera
     matrix into OpenCV's `cv2.solvePnP` function.
   - This function "solves" the 3D-to-2D perspective problem and gives
     us a `rotation_vector` and `translation_vector`.

5. Get Rotation Matrix:
   - We convert the `rotation_vector` into a full 3x3 `rotation_matrix`
     using `cv2.Rodrigues`.

6. Extract Euler Angles (Yaw, Pitch, Roll):
   - We decompose the `rotation_matrix` into the intuitive Yaw, Pitch,
     and Roll angles, using a function that safely handles the "gimbal
     lock" singularity.

7. State Machine (Gesture Detection):
   - This is our custom logic. It uses the new, accurate angles
     to detect "Quick" vs. "Slow" gestures and filters out "ticks."

8. Mouse Control via Head Movement:
   - Head yaw (left/right) and pitch (up/down) control mouse cursor
   - Calibration sets the neutral head position
   - Smooth movement with deadzone for stability

Press 'v' to toggle cube visualization on/off
Press 'k' to calibrate head gesture neutral position
Press F7 to toggle mouse control
Press F8 to toggle OptiKey data sending
=============================================================================
"""

# =============================================================================
# STUDENT ADAPTIVE GESTURE SYSTEM CONFIGURATION
# =============================================================================
# --- Thresholds (in degrees) ---
GESTURE_THRESHOLD_YAW = 15.0    # For LEFT/RIGHT
GESTURE_THRESHOLD_PITCH = 10.0 # For UP/DOWN

NEUTRAL_THRESHOLD_YAW = 5.0
NEUTRAL_THRESHOLD_PITCH = 5.0

# --- Timing (in seconds) ---
CALIBRATION_DURATION = 3.0  # How long to calibrate
QUICK_GESTURE_TIME = 0.8    # Gestures shorter than this are "quick"
GESTURE_COOLDOWN = 1.5      # Wait this long after a gesture
GESTURE_TIMEOUT = 6.0       # Cancel gesture if 'AWAY' for this long

# --- Persistence (in frames) ---
GESTURE_PERSISTENCE_FRAMES = 5 # Ignore "ticks" shorter than this
# =============================================================================

# =============================================================================
# HEAD-BASED MOUSE CONTROL CONFIGURATION
# =============================================================================
# Mouse sensitivity (degrees of head movement per pixel)
MOUSE_SENSITIVITY_YAW = 30.0    # Degrees of yaw for full screen width
MOUSE_SENSITIVITY_PITCH = 20.0  # Degrees of pitch for full screen height

# Deadzone to prevent jitter when head is neutral
MOUSE_DEADZONE_YAW = 2.0    # Degrees
MOUSE_DEADZONE_PITCH = 2.0  # Degrees

# Smoothing for mouse movement
MOUSE_SMOOTH_BUFFER_SIZE = 5
# =============================================================================

# Screen and mouse control setup
MONITOR_WIDTH, MONITOR_HEIGHT = pyautogui.size()
CENTER_X = MONITOR_WIDTH // 2
CENTER_Y = MONITOR_HEIGHT // 2

# OptiKey UDP Configuration
OPTIKEY_UDP_IP = "127.0.0.1"  # localhost
OPTIKEY_UDP_PORT = 5055       # MediaPipe port in OptiKey
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Global variables
mouse_control_enabled = False
optikey_enabled = True  # Send data to OptiKey

# Orbit camera state
orbit_yaw = -151.0
orbit_pitch = 0.0
orbit_radius = 1500.0
orbit_fov_deg = 50.0

# Debug world freeze
debug_world_frozen = False
orbit_pivot_frozen = None

# Mouse control
mouse_target = [CENTER_X, CENTER_Y]
mouse_lock = threading.Lock()
mouse_position_buffer = deque(maxlen=MOUSE_SMOOTH_BUFFER_SIZE)

# Head-based calibration offsets
calibration_offset_yaw = 0.0
calibration_offset_pitch = 0.0
head_calibrated = False

# Buffers for smoothing
head_angles_buffer = deque(maxlen=5)

# Reference matrices
R_ref_nose = [None]

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,  # Don't need iris landmarks
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Nose landmark indices
nose_indices = [4, 45, 275, 220, 440, 1, 5, 51, 281, 44, 274, 241,
                461, 125, 354, 218, 438, 195, 167, 393, 165, 391, 3, 248]

# Mouth landmark indices for open/close detection
mouth_vertical_indices = [13, 14]  # Top and bottom of mouth opening

# Enhanced features
fps_counter = deque(maxlen=30)
last_time = time.time()
high_contrast_mode = False
show_cube = True  # Toggle 3D cube visualization

# Mouth state tracking
last_mouth_state = False

# --- Student Gesture State Machine ---
gesture_state = 'NEUTRAL'  # States: 'NEUTRAL', 'CALIBRATING', 'AWAY', 'COOLDOWN'
gesture_start_time = 0.0
gesture_direction = None
neutral_yaw = 0.0        # The student's calibrated center yaw (in degrees)
neutral_pitch = 0.0      # The student's calibrated center pitch (in degrees)
calibration_start_time = 0.0
calibration_readings_yaw = []
calibration_readings_pitch = []
last_gesture_time = 0.0
last_gesture_detected = "" # To display on screen
gesture_persistence_counter = 0
gesture_timeout_counter = 0
GESTURE_TIMEOUT_THRESHOLD = 3  # Auto-recalibrate after this many timeouts
# -------------------------------------

# Configuration directory
def get_config_path():
    """Get configuration directory path"""
    path = Path.home() / ".headtracker"
    path.mkdir(parents=True, exist_ok=True)
    return path

# Screen position file
screen_position_file = get_config_path() / "screen_position.txt"

def write_screen_position(x, y):
    """Write screen position to file"""
    try:
        with open(screen_position_file, 'w') as f:
            f.write(f"{x},{y}\n")
    except:
        pass

def send_to_optikey(x, y):
    """Send screen position to OptiKey via UDP"""
    try:
        message = f"{x},{y}"
        udp_socket.sendto(message.encode(), (OPTIKEY_UDP_IP, OPTIKEY_UDP_PORT))
    except Exception as e:
        pass  # Silently fail if OptiKey is not running

# Helper functions
def _rot_x(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0],
                     [0, ca, -sa],
                     [0, sa,  ca]], dtype=float)

def _rot_y(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ ca, 0, sa],
                     [  0, 1,  0],
                     [-sa, 0, ca]], dtype=float)

def _normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v

def _focal_px(width, fov_deg):
    return 0.5 * width / math.tan(math.radians(fov_deg) * 0.5)

def compute_scale(points_3d):
    """Compute scale from 3D points"""
    if len(points_3d) < 2:
        return 1.0
    distances = []
    for i in range(len(points_3d)):
        for j in range(i + 1, len(points_3d)):
            dist = np.linalg.norm(points_3d[i] - points_3d[j])
            if dist > 0:
                distances.append(dist)
    return np.mean(distances) if distances else 1.0

def convert_head_to_screen_coordinates(head_yaw, head_pitch):
    """
    Convert head angles to screen coordinates for mouse control.
    Uses calibration offsets and applies sensitivity and deadzone.

    Args:
        head_yaw: Head yaw angle in degrees (positive = right)
        head_pitch: Head pitch angle in degrees (positive = up)

    Returns:
        (x, y): Screen coordinates
    """
    global calibration_offset_yaw, calibration_offset_pitch

    # Apply calibration offsets
    relative_yaw = head_yaw - calibration_offset_yaw
    relative_pitch = head_pitch - calibration_offset_pitch

    # Apply deadzone
    if abs(relative_yaw) < MOUSE_DEADZONE_YAW:
        relative_yaw = 0
    if abs(relative_pitch) < MOUSE_DEADZONE_PITCH:
        relative_pitch = 0

    # Convert to screen coordinates
    # Yaw: right = positive X, left = negative X
    x = CENTER_X + (relative_yaw / MOUSE_SENSITIVITY_YAW) * MONITOR_WIDTH

    # Pitch: up = negative Y (top of screen), down = positive Y (bottom)
    y = CENTER_Y - (relative_pitch / MOUSE_SENSITIVITY_PITCH) * MONITOR_HEIGHT

    # Clamp to screen bounds
    x = np.clip(x, 0, MONITOR_WIDTH - 1)
    y = np.clip(y, 0, MONITOR_HEIGHT - 1)

    return int(x), int(y)

def smooth_mouse_position(new_x, new_y):
    """
    Smooth mouse position using a buffer to reduce jitter.

    Args:
        new_x, new_y: New mouse position

    Returns:
        (smoothed_x, smoothed_y): Smoothed position
    """
    mouse_position_buffer.append((new_x, new_y))

    if len(mouse_position_buffer) == 0:
        return new_x, new_y

    # Calculate average position
    avg_x = int(np.mean([pos[0] for pos in mouse_position_buffer]))
    avg_y = int(np.mean([pos[1] for pos in mouse_position_buffer]))

    return avg_x, avg_y

def stabilize_rotation_matrix(R_cur, R_ref_container):
    """Stabilize rotation matrix to prevent flipping"""
    if R_ref_container[0] is not None:
        for i in range(3):
            if np.dot(R_cur[:, i], R_ref_container[0][:, i]) < 0:
                R_cur[:, i] = -R_cur[:, i]
    R_ref_container[0] = R_cur.copy()
    return R_cur

def update_orbit_from_keys():
    """Keyboard orbit controls"""
    global orbit_yaw, orbit_pitch, orbit_radius
    speed_yaw = 0.02
    speed_pitch = 0.02
    speed_zoom = 50

    if keyboard.is_pressed('j'):
        orbit_yaw -= speed_yaw
    if keyboard.is_pressed('l'):
        orbit_yaw += speed_yaw
    if keyboard.is_pressed('i'):
        orbit_pitch -= speed_pitch
    if keyboard.is_pressed('k'):
        orbit_pitch += speed_pitch
    if keyboard.is_pressed('['):
        orbit_radius += speed_zoom
    if keyboard.is_pressed(']'):
        orbit_radius = max(100, orbit_radius - speed_zoom)
    if keyboard.is_pressed('r'):
        orbit_yaw = -151.0
        orbit_pitch = 0.0
        orbit_radius = 1500.0

def detect_mouth_open(face_landmarks, w, h):
    """
    Detect if mouth is open based on vertical distance.
    Returns: (is_open, ratio)
    """
    try:
        # Simple vertical mouth opening detection
        # Using upper and lower lip landmarks
        upper_lip = face_landmarks[13]  # Upper lip center
        lower_lip = face_landmarks[14]  # Lower lip center

        # Calculate vertical distance
        mouth_height = abs(upper_lip.y - lower_lip.y) * h

        # Get mouth width for normalization
        left_corner = face_landmarks[61]
        right_corner = face_landmarks[291]
        mouth_width = abs(right_corner.x - left_corner.x) * w

        if mouth_width == 0:
            return False, 0.0

        # Calculate ratio
        mouth_ratio = mouth_height / mouth_width

        # Threshold for "open" (tune this value)
        is_open = mouth_ratio > 0.3

        return is_open, mouth_ratio
    except Exception as e:
        return False, 0.0

def rotation_matrix_to_euler_angles(R):
    """
    Convert rotation matrix to Euler angles (yaw, pitch, roll) in degrees.
    Handles gimbal lock gracefully.
    """
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0

    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)

def get_head_pose(face_landmarks, w, h):
    """
    Calculate head pose (yaw, pitch, roll) using solvePnP.
    Returns: (yaw, pitch, roll, head_center, rotation_matrix, nose_points_3d)
    """
    # 3D model points (canonical face model)
    model_points = np.array([
        (0.0, 0.0, 0.0),           # Nose tip (4)
        (0.0, -63.6, -12.5),       # Chin (152)
        (-43.3, 32.7, -26.0),      # Left eye left corner (33)
        (43.3, 32.7, -26.0),       # Right eye right corner (263)
        (-28.9, -28.9, -24.1),     # Left Mouth corner (61)
        (28.9, -28.9, -24.1)       # Right mouth corner (291)
    ], dtype=np.float64)

    # Corresponding 2D image points
    image_points = np.array([
        (face_landmarks[4].x * w, face_landmarks[4].y * h),      # Nose tip
        (face_landmarks[152].x * w, face_landmarks[152].y * h),  # Chin
        (face_landmarks[33].x * w, face_landmarks[33].y * h),    # Left eye corner
        (face_landmarks[263].x * w, face_landmarks[263].y * h),  # Right eye corner
        (face_landmarks[61].x * w, face_landmarks[61].y * h),    # Left mouth corner
        (face_landmarks[291].x * w, face_landmarks[291].y * h)   # Right mouth corner
    ], dtype=np.float64)

    # Camera matrix
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    # Assume no lens distortion
    dist_coeffs = np.zeros((4, 1))

    # Solve PnP
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return 0, 0, 0, np.zeros(3), np.eye(3), []

    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # Extract Euler angles
    yaw, pitch, roll = rotation_matrix_to_euler_angles(rotation_matrix)

    # Get head center (nose tip in 3D world coordinates)
    nose_tip_2d = image_points[0]
    nose_points_3d = [[face_landmarks[idx].x * w,
                       face_landmarks[idx].y * h,
                       face_landmarks[idx].z * w] for idx in nose_indices]
    head_center = np.mean(nose_points_3d, axis=0)

    return yaw, pitch, roll, head_center, rotation_matrix, nose_points_3d

def get_head_direction(yaw_deg, pitch_deg, threshold=10.0):
    """
    Get head direction as a string based on yaw and pitch angles.

    Args:
        yaw_deg: Yaw angle in degrees
        pitch_deg: Pitch angle in degrees
        threshold: Threshold for direction detection

    Returns:
        String: "LEFT", "RIGHT", "UP", "DOWN", or "CENTER"
    """
    if abs(yaw_deg) > abs(pitch_deg):
        if yaw_deg > threshold:
            return "RIGHT"
        elif yaw_deg < -threshold:
            return "LEFT"
    else:
        if pitch_deg > threshold:
            return "UP"
        elif pitch_deg < -threshold:
            return "DOWN"

    return "CENTER"

def draw_wireframe_cube(frame, center, R, size=80):
    """Draw a wireframe cube on the frame to visualize head orientation"""
    # Define cube vertices
    half_size = size / 2
    vertices = np.array([
        [-half_size, -half_size, -half_size],
        [half_size, -half_size, -half_size],
        [half_size, half_size, -half_size],
        [-half_size, half_size, -half_size],
        [-half_size, -half_size, half_size],
        [half_size, -half_size, half_size],
        [half_size, half_size, half_size],
        [-half_size, half_size, half_size]
    ])

    # Rotate and translate vertices
    rotated_vertices = []
    for v in vertices:
        rotated = R @ v + center
        rotated_vertices.append(rotated[:2].astype(int))

    # Draw cube edges
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Back face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Front face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting edges
    ]

    for edge in edges:
        pt1 = tuple(rotated_vertices[edge[0]])
        pt2 = tuple(rotated_vertices[edge[1]])
        cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

def draw_coordinate_axes(frame, center, R, length=100):
    """Draw coordinate axes to show head orientation"""
    # Define axis endpoints
    axes = np.array([
        [length, 0, 0],  # X-axis (red) - right
        [0, length, 0],  # Y-axis (green) - down
        [0, 0, length]   # Z-axis (blue) - forward
    ])

    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR

    origin_2d = center[:2].astype(int)

    for i, axis in enumerate(axes):
        rotated = R @ axis + center
        endpoint_2d = rotated[:2].astype(int)
        cv2.line(frame, tuple(origin_2d), tuple(endpoint_2d), colors[i], 3)

def draw_direction_arrow(frame, direction, center_x, center_y, length=40, color=(255, 255, 255)):
    """Draw an arrow pointing in the given direction"""
    if direction is None:
        return

    arrow_map = {
        "LEFT": (-length, 0),
        "RIGHT": (length, 0),
        "UP": (0, -length),
        "DOWN": (0, length),
        "CENTER": (0, 0)
    }

    if direction not in arrow_map:
        return

    dx, dy = arrow_map[direction]
    end_x = center_x + dx
    end_y = center_y + dy

    if direction != "CENTER":
        cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y), color, 3, tipLength=0.3)
    else:
        cv2.circle(frame, (center_x, center_y), 10, color, -1)

def draw_metrics(frame, head_direction=None,
                head_yaw=0, head_pitch=0, head_roll=0,
                gesture_detected="", gesture_state="NEUTRAL",
                mouth_open=False, mouth_ratio=0.0,
                mouse_enabled=False, optikey_enabled=False,
                head_calibrated=False):
    """
    Draw all metrics and status information on the frame.
    """
    h, w = frame.shape[:2]

    # Calculate FPS
    global last_time, fps_counter
    current_time = time.time()
    fps = 1.0 / (current_time - last_time + 1e-6)
    last_time = current_time
    fps_counter.append(fps)
    avg_fps = np.mean(fps_counter)

    # Status colors
    color_active = (0, 255, 0)
    color_inactive = (128, 128, 128)

    # Top-left: Status information
    y_offset = 30
    line_height = 25

    # FPS
    cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y_offset += line_height

    # Mouse control status
    mouse_color = color_active if mouse_enabled else color_inactive
    cv2.putText(frame, f"Mouse: {'ON' if mouse_enabled else 'OFF'} (F7)",
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mouse_color, 2)
    y_offset += line_height

    # OptiKey status
    optikey_color = color_active if optikey_enabled else color_inactive
    cv2.putText(frame, f"OptiKey: {'ON' if optikey_enabled else 'OFF'} (F8)",
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, optikey_color, 2)
    y_offset += line_height

    # Calibration status
    calib_color = color_active if head_calibrated else (0, 0, 255)
    calib_text = "Calibrated" if head_calibrated else "Not Calibrated (Press 'M')"
    cv2.putText(frame, f"Head: {calib_text}",
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, calib_color, 2)
    y_offset += line_height

    # Head angles
    cv2.putText(frame, f"Yaw: {head_yaw:.1f}  Pitch: {head_pitch:.1f}  Roll: {head_roll:.1f}",
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y_offset += line_height

    # Head direction
    if head_direction:
        cv2.putText(frame, f"Head Direction: {head_direction}",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += line_height

    # Mouth state
    mouth_color = (0, 255, 0) if mouth_open else (128, 128, 128)
    cv2.putText(frame, f"Mouth: {'OPEN' if mouth_open else 'CLOSED'} ({mouth_ratio:.2f})",
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mouth_color, 2)
    y_offset += line_height

    # Gesture state
    gesture_colors = {
        'NEUTRAL': (255, 255, 255),
        'CALIBRATING': (0, 255, 255),
        'AWAY': (0, 165, 255),
        'COOLDOWN': (128, 128, 128)
    }
    gesture_color = gesture_colors.get(gesture_state, (255, 255, 255))
    cv2.putText(frame, f"Gesture State: {gesture_state}",
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, gesture_color, 2)
    y_offset += line_height

    # Last gesture detected
    if gesture_detected:
        cv2.putText(frame, f"Gesture: {gesture_detected}",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += line_height

    # Center: Head direction arrow
    center_x, center_y = w // 2, h - 80
    if head_direction:
        draw_direction_arrow(frame, head_direction, center_x, center_y, length=50, color=(0, 255, 255))

    # Bottom: Instructions
    instructions = [
        "Controls: 'M'=Calibrate Mouse | 'K'=Calibrate Gestures | 'Q'=Quit",
        "          'V'=Toggle Cube | 'H'=High Contrast | F7=Mouse | F8=OptiKey"
    ]

    y_pos = h - 50
    for instruction in instructions:
        cv2.putText(frame, instruction, (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_pos += 20

def save_calibration(filename="calibration.json"):
    """Save calibration data to file"""
    global calibration_offset_yaw, calibration_offset_pitch
    global neutral_yaw, neutral_pitch

    calibration_data = {
        "calibration_offset_yaw": calibration_offset_yaw,
        "calibration_offset_pitch": calibration_offset_pitch,
        "neutral_yaw": neutral_yaw,
        "neutral_pitch": neutral_pitch,
        "timestamp": datetime.now().isoformat()
    }

    config_path = get_config_path()
    filepath = config_path / filename

    try:
        with open(filepath, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        print(f"[Save] Calibration saved to {filepath}")
    except Exception as e:
        print(f"[Error] Failed to save calibration: {e}")

def load_calibration(filename="calibration.json"):
    """Load calibration data from file"""
    global calibration_offset_yaw, calibration_offset_pitch
    global neutral_yaw, neutral_pitch, head_calibrated

    config_path = get_config_path()
    filepath = config_path / filename

    try:
        with open(filepath, 'r') as f:
            calibration_data = json.load(f)

        calibration_offset_yaw = calibration_data.get("calibration_offset_yaw", 0)
        calibration_offset_pitch = calibration_data.get("calibration_offset_pitch", 0)
        neutral_yaw = calibration_data.get("neutral_yaw", 0)
        neutral_pitch = calibration_data.get("neutral_pitch", 0)
        head_calibrated = True

        print(f"[Load] Calibration loaded from {filepath}")
        print(f"  Mouse: Yaw={calibration_offset_yaw:.1f}, Pitch={calibration_offset_pitch:.1f}")
        print(f"  Gesture: Yaw={neutral_yaw:.1f}, Pitch={neutral_pitch:.1f}")
    except FileNotFoundError:
        print(f"[Load] No calibration file found at {filepath}")
    except Exception as e:
        print(f"[Error] Failed to load calibration: {e}")

def main():
    """Main tracking loop"""
    global mouse_control_enabled, optikey_enabled
    global calibration_offset_yaw, calibration_offset_pitch, head_calibrated
    global gesture_state, gesture_start_time, gesture_direction
    global neutral_yaw, neutral_pitch
    global calibration_start_time, calibration_readings_yaw, calibration_readings_pitch
    global last_gesture_time, last_gesture_detected
    global gesture_persistence_counter, gesture_timeout_counter
    global high_contrast_mode, show_cube
    global last_mouth_state

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    print("Head Tracker Started!")
    print("=" * 60)
    print("Controls:")
    print("  M - Calibrate mouse control (look at center of screen)")
    print("  K - Calibrate head gestures (hold neutral pose)")
    print("  F7 - Toggle mouse control")
    print("  F8 - Toggle OptiKey data sending")
    print("  V - Toggle 3D cube visualization")
    print("  H - Toggle high contrast mode")
    print("  S - Save calibration")
    print("  L - Load calibration")
    print("  Q - Quit")
    print("=" * 60)

    # Attempt to load previous calibration
    load_calibration()

    # Variables for head direction
    raw_head_direction = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark

            # Get head pose
            head_yaw, head_pitch, head_roll, head_center, R_final, nose_points_3d = get_head_pose(face_landmarks, w, h)

            # Stabilize rotation matrix
            R_final = stabilize_rotation_matrix(R_final, R_ref_nose)

            # Get head direction
            raw_head_direction = get_head_direction(head_yaw, head_pitch, threshold=10.0)

            # Detect mouth open
            mouth_open, mouth_ratio = detect_mouth_open(face_landmarks, w, h)

            # Print mouth state changes to command line
            global last_mouth_state
            if mouth_open != last_mouth_state:
                print(f"\n>>> MOUTH: {'OPEN' if mouth_open else 'CLOSED'} (Ratio: {mouth_ratio:.2f})\n")
                last_mouth_state = mouth_open

            # --- STUDENT ADAPTIVE GESTURE LOGIC ---
            current_time = time.time()

            if gesture_state == 'CALIBRATING':
                elapsed = current_time - calibration_start_time
                if elapsed < CALIBRATION_DURATION:
                    # Collect readings
                    calibration_readings_yaw.append(head_yaw)
                    calibration_readings_pitch.append(head_pitch)
                else:
                    # Calibration complete
                    neutral_yaw = np.median(calibration_readings_yaw)
                    neutral_pitch = np.median(calibration_readings_pitch)
                    print("\n" + "="*60)
                    print(f">>> GESTURE CALIBRATION COMPLETE!")
                    print(f">>> Neutral Yaw: {neutral_yaw:.2f}°")
                    print(f">>> Neutral Pitch: {neutral_pitch:.2f}°")
                    print("="*60 + "\n")
                    gesture_state = 'NEUTRAL'
                    last_gesture_detected = "Calibration Complete!"

            elif gesture_state == 'NEUTRAL':
                relative_yaw = head_yaw - neutral_yaw
                relative_pitch = head_pitch - neutral_pitch

                # Check if moved away from neutral
                direction = None
                if abs(relative_yaw) > abs(relative_pitch):
                    if relative_yaw > GESTURE_THRESHOLD_YAW:
                        direction = "RIGHT"
                    elif relative_yaw < -GESTURE_THRESHOLD_YAW:
                        direction = "LEFT"
                else:
                    if relative_pitch > GESTURE_THRESHOLD_PITCH:
                        direction = "UP"
                    elif relative_pitch < -GESTURE_THRESHOLD_PITCH:
                        direction = "DOWN"

                if direction:
                    # NEW: Persistence check
                    if gesture_direction == direction:
                        gesture_persistence_counter += 1
                    else:
                        gesture_direction = direction
                        gesture_persistence_counter = 1

                    if gesture_persistence_counter >= GESTURE_PERSISTENCE_FRAMES:
                        gesture_state = 'AWAY'
                        gesture_start_time = current_time
                        last_gesture_detected = f"Moving {direction}..."
                else:
                    gesture_persistence_counter = 0

            elif gesture_state == 'AWAY':
                # Check for returning to NEUTRAL
                relative_yaw = head_yaw - neutral_yaw
                relative_pitch = head_pitch - neutral_pitch
                is_neutral_now = (abs(relative_yaw) < NEUTRAL_THRESHOLD_YAW and
                                  abs(relative_pitch) < NEUTRAL_THRESHOLD_PITCH)

                if is_neutral_now:
                    # GESTURE COMPLETED
                    time_elapsed = current_time - gesture_start_time

                    if time_elapsed < QUICK_GESTURE_TIME:
                        gesture_type = "QUICK"
                    else:
                        gesture_type = "SLOW"

                    last_gesture_detected = f"{gesture_type} {gesture_direction}"
                    print(f"\n>>> GESTURE COMPLETED: {gesture_type} {gesture_direction} (Duration: {time_elapsed:.2f}s)\n")

                    gesture_timeout_counter = 0
                    gesture_state = 'COOLDOWN'
                    last_gesture_time = current_time
                    gesture_direction = None

                elif (current_time - gesture_start_time) > GESTURE_TIMEOUT:
                    # GESTURE TIMEOUT
                    gesture_timeout_counter += 1
                    print(f"\n>>> GESTURE TIMEOUT: {gesture_direction} (held too long: {GESTURE_TIMEOUT}s)")
                    print(f">>> Timeout count: {gesture_timeout_counter}/{GESTURE_TIMEOUT_THRESHOLD}\n")
                    last_gesture_detected = "Gesture Failed (Timeout)"

                    if gesture_timeout_counter >= GESTURE_TIMEOUT_THRESHOLD:
                        print("\n" + "="*60)
                        print(">>> AUTO-RECALIBRATION TRIGGERED")
                        print(f">>> Too many consecutive timeouts ({gesture_timeout_counter})")
                        print(f">>> Starting automatic head gesture calibration...")
                        print("="*60 + "\n")
                        gesture_state = 'CALIBRATING'
                        calibration_start_time = current_time
                        calibration_readings_yaw = []
                        calibration_readings_pitch = []
                        last_gesture_detected = "AUTO-CALIBRATING..."
                        gesture_timeout_counter = 0
                    else:
                        gesture_state = 'COOLDOWN'
                        last_gesture_time = current_time
                        gesture_direction = None

            elif gesture_state == 'COOLDOWN':
                if current_time - last_gesture_time > GESTURE_COOLDOWN:
                    gesture_state = 'NEUTRAL'
                    if "Calibration" not in last_gesture_detected:
                        last_gesture_detected = ""

            # --- END: STUDENT ADAPTIVE GESTURE LOGIC ---

            # Draw 3D cube and axes on face
            if show_cube:
                try:
                    draw_wireframe_cube(frame, head_center, R_final, size=60)
                    draw_coordinate_axes(frame, head_center, R_final, length=80)
                except:
                    pass

            # Draw face mesh (simplified - just landmarks)
            for lm in face_landmarks:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 1, (128, 128, 128), -1)

            # Head-based mouse control
            if head_calibrated and (mouse_control_enabled or optikey_enabled):
                screen_x, screen_y = convert_head_to_screen_coordinates(head_yaw, head_pitch)

                # Smooth mouse position
                smooth_x, smooth_y = smooth_mouse_position(screen_x, screen_y)

                # Draw cursor position indicator on frame
                cursor_x = int((smooth_x / MONITOR_WIDTH) * w)
                cursor_y = int((smooth_y / MONITOR_HEIGHT) * h)
                cv2.circle(frame, (cursor_x, cursor_y), 15, (0, 255, 0), 2)
                cv2.line(frame, (cursor_x - 20, cursor_y), (cursor_x + 20, cursor_y), (0, 255, 0), 2)
                cv2.line(frame, (cursor_x, cursor_y - 20), (cursor_x, cursor_y + 20), (0, 255, 0), 2)

                # Update mouse if enabled
                if mouse_control_enabled:
                    with mouse_lock:
                        mouse_target[0] = smooth_x
                        mouse_target[1] = smooth_y
                    pyautogui.moveTo(smooth_x, smooth_y)

                # Send to OptiKey if enabled
                if optikey_enabled:
                    send_to_optikey(smooth_x, smooth_y)

                # Write screen position
                write_screen_position(smooth_x, smooth_y)

            # Draw metrics
            draw_metrics(frame, raw_head_direction,
                        head_yaw, head_pitch, head_roll,
                        last_gesture_detected, gesture_state,
                        mouth_open, mouth_ratio,
                        mouse_control_enabled, optikey_enabled,
                        head_calibrated)

            # Update orbit controls
            update_orbit_from_keys()
        else:
            # No face detected
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            draw_metrics(frame, None, 0, 0, 0, "No face detected", "NEUTRAL",
                        False, 0.0, mouse_control_enabled, optikey_enabled, head_calibrated)

        # Apply high contrast if enabled
        if high_contrast_mode:
            frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=20)

        cv2.imshow("Head Tracker", frame)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('m'):
            # Calibrate mouse control (set current head position as center)
            if results.multi_face_landmarks:
                calibration_offset_yaw = head_yaw
                calibration_offset_pitch = head_pitch
                head_calibrated = True
                print("\n" + "="*60)
                print(">>> MOUSE CALIBRATION COMPLETE!")
                print(f">>> Center Yaw: {calibration_offset_yaw:.2f}°")
                print(f">>> Center Pitch: {calibration_offset_pitch:.2f}°")
                print(">>> Current head position set as screen center")
                print("="*60 + "\n")
                # Clear mouse position buffer
                mouse_position_buffer.clear()
        elif key == ord('k'):
            # Calibrate head gestures
            if gesture_state != 'CALIBRATING':
                print("\n" + "="*60)
                print(">>> STARTING HEAD GESTURE CALIBRATION")
                print(f">>> Hold a neutral head pose for {CALIBRATION_DURATION} seconds...")
                print("="*60 + "\n")
                gesture_state = 'CALIBRATING'
                calibration_start_time = time.time()
                calibration_readings_yaw = []
                calibration_readings_pitch = []
                last_gesture_detected = "CALIBRATING..."
                gesture_timeout_counter = 0
            else:
                print(">>> Calibration already in progress...")
        elif key == ord('s'):
            save_calibration()
        elif key == ord('l'):
            load_calibration()
        elif key == ord('h'):
            high_contrast_mode = not high_contrast_mode
            print(f"High contrast: {'ON' if high_contrast_mode else 'OFF'}")
        elif key == ord('v'):
            show_cube = not show_cube
            print(f"3D Cube visualization: {'ON' if show_cube else 'OFF'}")

        # Function key handling
        if keyboard.is_pressed('f7'):
            if not mouse_control_enabled and not head_calibrated:
                print(">>> Please calibrate mouse control first (press 'M')")
            else:
                mouse_control_enabled = not mouse_control_enabled
                print(f"Mouse control: {'ON' if mouse_control_enabled else 'OFF'}")
            time.sleep(0.3)
        elif keyboard.is_pressed('f8'):
            if not optikey_enabled and not head_calibrated:
                print(">>> Please calibrate mouse control first (press 'M')")
            else:
                optikey_enabled = not optikey_enabled
                print(f"OptiKey control: {'ON' if optikey_enabled else 'OFF'}")
            time.sleep(0.3)

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\nHead Tracker closed")

if __name__ == "__main__":
    main()
