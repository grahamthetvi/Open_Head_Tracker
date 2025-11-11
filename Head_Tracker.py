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

# Disable PyAutoGUI fail-safe to prevent corner-triggered stops
# This is important for eye gaze tracking that controls the mouse
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0  # Remove delay between PyAutoGUI commands
import json
from datetime import datetime
from pathlib import Path
import socket  # For UDP communication with OptiKey

# Kalman filter for smooth tracking (Option 3A)
try:
    import filterpy
    KALMAN_AVAILABLE = True
except ImportError:
    KALMAN_AVAILABLE = False
    print("[Warning] filterpy not available. Install with: pip install filterpy")


"""
=============================================================================
HEAD TRACKING SYSTEM EXPLANATION
=============================================================================

HOW HEAD DIRECTION IS DETERMINED: (NEW "PRO FIX" METHOD)

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

Press 'v' to toggle cube visualization on/off
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

# --- NEW: Persistence (in frames) ---
GESTURE_PERSISTENCE_FRAMES = 5 # Ignore "ticks" shorter than this
# =============================================================================


# Screen and mouse control setup
MONITOR_WIDTH, MONITOR_HEIGHT = pyautogui.size()
CENTER_X = MONITOR_WIDTH // 2
CENTER_Y = MONITOR_HEIGHT // 2

# OptiKey UDP Configuration
OPTIKEY_UDP_IP = "127.0.0.1"  # localhost
OPTIKEY_UDP_PORT = 5055       # MediaPipe port in OptiKey
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Global variables from original
mouse_control_enabled = False
optikey_enabled = True  # NEW: Send data to OptiKey
filter_length = 10
gaze_length = 350

# Orbit camera state
orbit_yaw = -151.0
orbit_pitch = 0.0
orbit_radius = 1500.0
orbit_fov_deg = 50.0

# Debug world freeze
debug_world_frozen = False
orbit_pivot_frozen = None

# Gaze markers
gaze_markers = []

# 3D monitor plane state
monitor_corners = None
monitor_center_w = None
monitor_normal_w = None
units_per_cm = None

# Mouse control
mouse_target = [CENTER_X, CENTER_Y]
mouse_lock = threading.Lock()


# Calibration offsets
calibration_offset_yaw = 0
calibration_offset_pitch = 0

# Calibration step
calib_step = 0

# Buffers for smoothing
combined_gaze_directions = deque(maxlen=filter_length)
head_angles_buffer = deque(maxlen=5)  # Smaller buffer for head tracking

# ROI-based eye processing buffers (Option 2)
left_eye_ratio_buffer = deque(maxlen=5)  # Buffer for left eye ratios
right_eye_ratio_buffer = deque(maxlen=5)  # Buffer for right eye ratios
use_roi_processing = True  # Toggle for ROI-based processing

# Hybrid approach (Option 3A) - OpenCV + MediaPipe
use_hybrid_processing = False  # Toggle for hybrid processing
left_eye_kalman = None
right_eye_kalman = None

# Reference matrices
R_ref_nose = [None]
R_ref_forehead = [None]
calibration_nose_scale = None

# Eye sphere tracking from original
left_sphere_locked = False
right_sphere_locked = False
left_sphere_local_offset = None
right_sphere_local_offset = None
left_calibration_nose_scale = None
right_calibration_nose_scale = None

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Nose landmark indices
nose_indices = [4, 45, 275, 220, 440, 1, 5, 51, 281, 44, 274, 241, 
                461, 125, 354, 218, 438, 195, 167, 393, 165, 391, 3, 248]

# Eye landmark indices for enhanced features
left_eye_indices = [33, 160, 158, 133, 153, 144, 163, 7]
right_eye_indices = [362, 385, 387, 263, 373, 374, 380, 249]
left_iris_indices = [468, 469, 470, 471, 472]
right_iris_indices = [473, 474, 475, 476, 477]

# --- NEWLY ADDED LANDMARK LISTS ---
# NEW: 71 Landmarks for the LEFT eye contour
LEFT_EYE_LANDMARKS_71 = [
    33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173,
    263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388,
    466, 263, 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160,
    161, 246, 247, 30, 29, 27, 28, 56, 190, 130, 25, 110, 24, 23, 22, 26, 112,
    243, 113, 225, 224, 223, 222, 221, 189
]

# NEW: 71 Landmarks for the RIGHT eye contour
RIGHT_EYE_LANDMARKS_71 = [
    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384,
    398, 33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144,
    163, 7, 467, 255, 254, 253, 252, 256, 339, 463, 342, 445, 444, 443, 442,
    441, 413, 467, 260, 259, 257, 258, 286, 414, 359, 255, 339, 254, 253, 252,
    256, 341, 463, 342, 445, 444, 443, 442, 441, 413
]
# --- END OF NEW LISTS ---

LEFT_EYE_CORNERS = [33, 133]
LEFT_EYE_LIDS = [159, 145]
RIGHT_EYE_CORNERS = [362, 263]
RIGHT_EYE_LIDS = [386, 374]

# Mouth landmark indices for open/close detection
# Inner lips vertical landmarks
mouth_top_inner = [13, 14]  # Top inner lip center
mouth_bottom_inner = [13, 14]  # Bottom inner lip center
# Outer lips for better detection
mouth_top_outer = [0, 267, 269, 270, 409, 291]
mouth_bottom_outer = [17, 375, 321, 405, 314, 17]
# Simpler: just vertical mouth points
mouth_vertical_indices = [13, 14]  # Top and bottom of mouth opening

# Enhanced features
blink_threshold = 0.15
total_blinks = 0
eye_states = deque(maxlen=10)
fps_counter = deque(maxlen=30)
last_time = time.time()
high_contrast_mode = False
enable_logging = False
log_data = []
show_cube = True  # Toggle 3D cube visualization
detect_looking_up_enabled = False  # Toggle for looking up detection (off by default)

# Mouth state tracking for command prompt output
last_mouth_state = False  # Track previous mouth state to detect changes
last_looking_up_state = False
last_gaze_direction_spoken = "CENTER"

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
gesture_persistence_counter = 0 # NEW: For filtering "ticks"
gesture_timeout_counter = 0  # Track consecutive timeouts for auto-recalibration
GESTURE_TIMEOUT_THRESHOLD = 3  # Auto-recalibrate after this many timeouts
# -------------------------------------


# Configuration directory
def get_config_path():
    """Get configuration directory path"""
    path = Path.home() / ".eyetracker3d"
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

# Helper functions from original
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

# Functions from original code
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

def convert_gaze_to_screen_coordinates(gaze_direction, calibration_yaw=0, calibration_pitch=0):
    """Convert gaze direction to screen coordinates"""
    gaze_dir = _normalize(gaze_direction)
    
    # MediaPipe coordinate system:
    # X: right, Y: down (in image), Z: toward camera
    
    # Yaw (left/right): positive = right, negative = left
    yaw = math.atan2(gaze_dir[0], gaze_dir[2])
    
    # Pitch (up/down): MediaPipe Y is downward, so negate for intuitive pitch
    # Positive pitch = looking up, negative pitch = looking down
    pitch = -math.asin(np.clip(gaze_dir[1], -1, 1))
    
    yaw += calibration_yaw
    pitch += calibration_pitch
    
    # Screen coordinates: X increases right, Y increases down
    # Positive yaw (looking right) → increase X
    x = CENTER_X + (yaw / math.radians(30)) * CENTER_X
    
    # Positive pitch (looking up) → decrease Y (top of screen)
    # Negative pitch (looking down) → increase Y (bottom of screen)
    y = CENTER_Y - (pitch / math.radians(20)) * CENTER_Y
    
    x = np.clip(x, 0, MONITOR_WIDTH)
    y = np.clip(y, 0, MONITOR_HEIGHT)
    
    return int(x), int(y), yaw, pitch



def create_monitor_plane(head_center, R_final, face_landmarks, w, h, 
                         forward_hint=None, gaze_origin=None, gaze_dir=None):
    """Build a 60cm x 40cm plane 50cm in front of the face"""
    try:
        lm_chin = face_landmarks[152]
        lm_fore = face_landmarks[10]
        chin_w = np.array([lm_chin.x * w,  lm_chin.y * h,  lm_chin.z * w], dtype=float)
        fore_w = np.array([lm_fore.x * w,  lm_fore.y * h,  lm_fore.z * w], dtype=float)
        face_h_units = np.linalg.norm(fore_w - chin_w)
        upc = face_h_units / 15.0
    except Exception:
        upc = 5.0
    
    dist_cm = 50.0
    mon_w_cm, mon_h_cm = 60.0, 40.0
    half_w = (mon_w_cm * 0.5) * upc
    half_h = (mon_h_cm * 0.5) * upc

    head_forward = -R_final[:, 2]
    if forward_hint is not None:
        head_forward = forward_hint / np.linalg.norm(forward_hint)

    if gaze_origin is not None and gaze_dir is not None:
        gaze_dir = gaze_dir / np.linalg.norm(gaze_dir)
        plane_point = head_center + head_forward * (50.0 * upc)
        plane_normal = head_forward
        denom = np.dot(plane_normal, gaze_dir)
        if abs(denom) > 1e-6:
            t = np.dot(plane_normal, plane_point - gaze_origin) / denom
            center_w = gaze_origin + t * gaze_dir
        else:
            center_w = head_center + head_forward * (50.0 * upc)
    else:
        center_w = head_center + head_forward * (50.0 * upc)

    world_up = np.array([0, -1, 0], dtype=float)
    head_right = np.cross(world_up, head_forward)
    head_right /= np.linalg.norm(head_right) + 1e-9
    head_up = np.cross(head_forward, head_right)
    head_up /= np.linalg.norm(head_up) + 1e-9

    p0 = center_w - head_right * half_w - head_up * half_h
    p1 = center_w + head_right * half_w - head_up * half_h
    p2 = center_w + head_right * half_w + head_up * half_h
    p3 = center_w - head_right * half_w + head_up * half_h

    normal_w = head_forward / (np.linalg.norm(head_forward) + 1e-9)
    return [p0, p1, p2, p3], center_w, normal_w, upc

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

def calculate_eye_aspect_ratio(eye_landmarks):
    """Calculate eye aspect ratio for blink detection"""
    if len(eye_landmarks) < 6:
        return 0
    
    vertical_dist1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    vertical_dist2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    horizontal_dist = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    
    if horizontal_dist == 0:
        return 0
    
    ear = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)
    return ear

def detect_blink(face_landmarks, w, h):
    """Detect eye blinks using eye aspect ratio"""
    global total_blinks, eye_states
    
    # Get left eye landmarks
    left_eye = []
    for idx in left_eye_indices[:6]:
        lm = face_landmarks[idx]
        left_eye.append(np.array([lm.x * w, lm.y * h]))
    
    # Get right eye landmarks  
    right_eye = []
    for idx in right_eye_indices[:6]:
        lm = face_landmarks[idx]
        right_eye.append(np.array([lm.x * w, lm.y * h]))
    
    # Calculate EAR for both eyes
    left_ear = calculate_eye_aspect_ratio(np.array(left_eye))
    right_ear = calculate_eye_aspect_ratio(np.array(right_eye))
    avg_ear = (left_ear + right_ear) / 2.0
    
    # Detect blink
    is_blinking = avg_ear < blink_threshold
    eye_states.append(is_blinking)
    
    # Count blinks (rising edge detection)
    if len(eye_states) >= 2:
        if not eye_states[-2] and eye_states[-1]:
            total_blinks += 1
            return True, avg_ear
    
    return False, avg_ear

def detect_looking_up(face_landmarks, w, h):
    """
    Detect if eyes are looking up based on iris position relative to eye landmarks.
    Returns: (is_looking_up, avg_ratio)
    """
    try:
        # Use multiple landmarks to get better eye boundary detection
        # Left eye: top landmarks
        left_top_landmarks = [159, 160, 161, 158]
        left_bottom_landmarks = [145, 144, 153, 154]
        
        # Right eye: top landmarks
        right_top_landmarks = [386, 387, 388, 385]
        right_bottom_landmarks = [374, 373, 380, 381]
        
        # Get average Y positions for eye boundaries
        left_eye_top_y = np.mean([face_landmarks[idx].y for idx in left_top_landmarks])
        left_eye_bottom_y = np.mean([face_landmarks[idx].y for idx in left_bottom_landmarks])
        
        right_eye_top_y = np.mean([face_landmarks[idx].y for idx in right_top_landmarks])
        right_eye_bottom_y = np.mean([face_landmarks[idx].y for idx in right_bottom_landmarks])
        
        # Get iris centers (average of iris landmarks)
        left_iris_y = np.mean([face_landmarks[idx].y for idx in left_iris_indices])
        right_iris_y = np.mean([face_landmarks[idx].y for idx in right_iris_indices])
        
        # Calculate vertical position of iris within eye (0 = top, 1 = bottom)
        left_eye_height = left_eye_bottom_y - left_eye_top_y
        right_eye_height = right_eye_bottom_y - right_eye_top_y
        
        # Prevent division by zero
        if left_eye_height < 0.001 or right_eye_height < 0.001:
            return False, 0.5
        
        # Calculate ratio: how far down is the iris from the top?
        # (lower value = iris is higher = looking up)
        left_ratio = (left_iris_y - left_eye_top_y) / left_eye_height
        right_ratio = (right_iris_y - right_eye_top_y) / right_eye_height
        
        # Average the ratios
        avg_ratio = (left_ratio + right_ratio) / 2.0

        # Invert so 1.0 is top (looking up), 0.0 is bottom (looking down)
        avg_ratio = 1.0 - avg_ratio

        # Clamp ratio to valid range
        avg_ratio = np.clip(avg_ratio, 0.0, 1.0)
        
        # If iris is in upper portion of eye (ratio > 0.75), person is looking up
        # Higher threshold requires more extreme upward gaze
        is_looking_up = avg_ratio > 0.75
        
        return is_looking_up, avg_ratio
        
    except Exception as e:
        # If anything goes wrong, return safe defaults
        return False, 0.5


def detect_mouth_open(face_landmarks, w, h):
    """
    Detect if mouth is open based on vertical distance between lips.
    Returns: (is_mouth_open, mouth_ratio)
    """
    try:
        # Key mouth landmarks for vertical opening
        # Upper lip: landmarks on top of mouth
        upper_lip_top = [13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82]
        # Lower lip: landmarks on bottom of mouth  
        lower_lip_bottom = [17, 84, 181, 91, 146, 61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314]
        
        # Get a few key vertical points for simplicity
        # Top of upper lip
        upper_points = [13, 14]
        # Bottom of lower lip
        lower_points = [17, 18]
        
        # Also get mouth corners for width
        mouth_left = face_landmarks[61]
        mouth_right = face_landmarks[291]
        
        # Calculate average positions
        upper_y = np.mean([face_landmarks[idx].y for idx in upper_points])
        lower_y = np.mean([face_landmarks[idx].y for idx in lower_points])
        
        # Calculate mouth width (for normalization)
        mouth_width = np.sqrt((mouth_right.x - mouth_left.x)**2 + 
                              (mouth_right.y - mouth_left.y)**2)
        
        # Calculate mouth opening (vertical distance)
        mouth_opening = abs(lower_y - upper_y)
        
        # Prevent division by zero
        if mouth_width < 0.001:
            return False, 0.0
        
        # Calculate Mouth Aspect Ratio (MAR)
        # Ratio of vertical opening to horizontal width
        mouth_ratio = mouth_opening / mouth_width
        
        # Round up closed mouth values around 0.3 to 0.3375
        if 0.28 <= mouth_ratio <= 0.32:
            mouth_ratio = 0.3375
        
        # User-specific calibration:
        # Closed mouth: ~0.3375 (after rounding)
        # Open mouth: 0.5+
        # Threshold for "open" detection
        is_mouth_open = mouth_ratio > 0.5
        
        return is_mouth_open, mouth_ratio
        
    except Exception as e:
        # If anything goes wrong, return safe defaults
        return False, 0.0


def get_iris_position_ratio(face_landmarks, eye_corners, eye_lids, iris_indices, w, h,
                            eye_contour_indices=None):
    """
    Calculate iris position ratio within the eye boundaries.
    
    Uses 71 eye contour landmarks for improved accuracy when available,
    otherwise falls back to eye corners and lids.
    
    Args:
        face_landmarks: MediaPipe face landmarks
        eye_corners: List of 2 corner landmark indices (for fallback)
        eye_lids: List of 2 lid landmark indices (for fallback)
        iris_indices: List of iris landmark indices
        w, h: Frame width and height
        eye_contour_indices: Optional list of 71 eye contour landmark indices
                           (LEFT_EYE_LANDMARKS_71 or RIGHT_EYE_LANDMARKS_71)
    
    Returns:
        (horizontal_ratio, vertical_ratio) tuple, both in range [0.0, 1.0]
    """
    try:
        iris_center_x = np.mean([face_landmarks[idx].x for idx in iris_indices])
        iris_center_y = np.mean([face_landmarks[idx].y for idx in iris_indices])

        # IMPROVED: Use 71 eye contour landmarks if available for better boundary detection
        if eye_contour_indices is not None and len(eye_contour_indices) > 0:
            # Calculate eye boundary from all contour landmarks
            contour_x = [face_landmarks[idx].x for idx in eye_contour_indices]
            contour_y = [face_landmarks[idx].y for idx in eye_contour_indices]
            
            eye_left = min(contour_x)
            eye_right = max(contour_x)
            eye_top = min(contour_y)
            eye_bottom = max(contour_y)
            
            eye_width = eye_right - eye_left
            eye_height = eye_bottom - eye_top
            
            if eye_width > 0:
                horizontal_ratio = (iris_center_x - eye_left) / eye_width
            else:
                horizontal_ratio = 0.5
            
            if eye_height > 0:
                vertical_ratio = (iris_center_y - eye_top) / eye_height
            else:
                vertical_ratio = 0.5
        else:
            # FALLBACK: Use original method with corners and lids
            left_corner = face_landmarks[eye_corners[0]]
            right_corner = face_landmarks[eye_corners[1]]
            
            top_lid = face_landmarks[eye_lids[0]]
            bottom_lid = face_landmarks[eye_lids[1]]
            
            eye_width = right_corner.x - left_corner.x
            if eye_width > 0:
                horizontal_ratio = (iris_center_x - left_corner.x) / eye_width
            else:
                horizontal_ratio = 0.5
            
            eye_height = bottom_lid.y - top_lid.y
            if eye_height > 0:
                vertical_ratio = (iris_center_y - top_lid.y) / eye_height
            else:
                vertical_ratio = 0.5

        # Invert vertical ratio so 1.0 = top (looking up), 0.0 = bottom (looking down)
        vertical_ratio = 1.0 - vertical_ratio

        horizontal_ratio = np.clip(horizontal_ratio, 0.0, 1.0)
        vertical_ratio = np.clip(vertical_ratio, 0.0, 1.0)

        return horizontal_ratio, vertical_ratio

    except Exception:
        return 0.5, 0.5


def extract_eye_roi_info(face_landmarks, eye_contour_indices, w, h, padding_ratio=0.2):
    """
    Extract eye ROI information including bounding box and normalized coordinates.
    
    Args:
        face_landmarks: MediaPipe face landmarks
        eye_contour_indices: List of eye contour landmark indices
        w, h: Frame width and height
        padding_ratio: Padding around eye (as ratio of eye size)
    
    Returns:
        dict with keys: 'bbox' (x, y, width, height), 'center', 'scale', 
                       'normalized_landmarks' (list of normalized coordinates)
    """
    try:
        # Get all contour points
        contour_points = np.array([
            [face_landmarks[idx].x * w, face_landmarks[idx].y * h]
            for idx in eye_contour_indices
        ])
        
        # Calculate bounding box
        min_x = int(np.min(contour_points[:, 0]))
        max_x = int(np.max(contour_points[:, 0]))
        min_y = int(np.min(contour_points[:, 1]))
        max_y = int(np.max(contour_points[:, 1]))
        
        # Add padding
        eye_width = max_x - min_x
        eye_height = max_y - min_y
        padding_x = int(eye_width * padding_ratio)
        padding_y = int(eye_height * padding_ratio)
        
        # Clamp to frame boundaries
        roi_x = max(0, min_x - padding_x)
        roi_y = max(0, min_y - padding_y)
        roi_width = min(w - roi_x, max_x + padding_x - roi_x)
        roi_height = min(h - roi_y, max_y + padding_y - roi_y)
        
        # Calculate center and scale for normalization
        center_x = roi_x + roi_width / 2.0
        center_y = roi_y + roi_height / 2.0
        scale = max(roi_width, roi_height) / 2.0
        
        # Normalize landmarks relative to ROI center
        normalized_landmarks = []
        for point in contour_points:
            norm_x = (point[0] - center_x) / scale if scale > 0 else 0.0
            norm_y = (point[1] - center_y) / scale if scale > 0 else 0.0
            normalized_landmarks.append([norm_x, norm_y])
        
        return {
            'bbox': (roi_x, roi_y, roi_width, roi_height),
            'center': (center_x, center_y),
            'scale': scale,
            'normalized_landmarks': np.array(normalized_landmarks)
        }
    except Exception:
        return None


def get_iris_position_ratio_roi(face_landmarks, iris_indices, eye_roi_info, w, h):
    """
    Calculate iris position ratio within eye ROI using normalized coordinates.
    This provides better accuracy when head is rotated or eyes are asymmetric.
    
    Args:
        face_landmarks: MediaPipe face landmarks
        iris_indices: List of iris landmark indices
        eye_roi_info: ROI info dict from extract_eye_roi_info()
        w, h: Frame width and height
    
    Returns:
        (horizontal_ratio, vertical_ratio) tuple
    """
    try:
        if eye_roi_info is None:
            return 0.5, 0.5
        
        # Get iris center in pixel coordinates
        iris_center_x = np.mean([face_landmarks[idx].x * w for idx in iris_indices])
        iris_center_y = np.mean([face_landmarks[idx].y * h for idx in iris_indices])
        
        # Normalize iris position relative to ROI center
        center_x, center_y = eye_roi_info['center']
        scale = eye_roi_info['scale']
        
        if scale <= 0:
            return 0.5, 0.5
        
        norm_iris_x = (iris_center_x - center_x) / scale
        norm_iris_y = (iris_center_y - center_y) / scale
        
        # Get normalized eye boundary from contour landmarks
        normalized_landmarks = eye_roi_info['normalized_landmarks']
        if len(normalized_landmarks) == 0:
            return 0.5, 0.5
        
        eye_left_norm = np.min(normalized_landmarks[:, 0])
        eye_right_norm = np.max(normalized_landmarks[:, 0])
        eye_top_norm = np.min(normalized_landmarks[:, 1])
        eye_bottom_norm = np.max(normalized_landmarks[:, 1])
        
        # Calculate ratios in normalized space
        eye_width_norm = eye_right_norm - eye_left_norm
        eye_height_norm = eye_bottom_norm - eye_top_norm
        
        if eye_width_norm > 0:
            horizontal_ratio = (norm_iris_x - eye_left_norm) / eye_width_norm
        else:
            horizontal_ratio = 0.5
        
        if eye_height_norm > 0:
            vertical_ratio = (norm_iris_y - eye_top_norm) / eye_height_norm
        else:
            vertical_ratio = 0.5
        
        # Invert vertical ratio (1.0 = top, 0.0 = bottom)
        vertical_ratio = 1.0 - vertical_ratio
        
        horizontal_ratio = np.clip(horizontal_ratio, 0.0, 1.0)
        vertical_ratio = np.clip(vertical_ratio, 0.0, 1.0)
        
        return horizontal_ratio, vertical_ratio
        
    except Exception:
        return 0.5, 0.5


def get_combined_gaze_direction(left_h_ratio, left_v_ratio, right_h_ratio, right_v_ratio,
                                h_threshold=0.3, v_threshold=0.25):
    """Determine gaze direction based on iris position ratios from both eyes."""
    avg_h_ratio = (left_h_ratio + right_h_ratio) / 2.0
    avg_v_ratio = (left_v_ratio + right_v_ratio) / 2.0

    directions = []

    if avg_h_ratio < (0.5 - h_threshold):
        directions.append("LEFT")
    elif avg_h_ratio > (0.5 + h_threshold):
        directions.append("RIGHT")

    if avg_v_ratio < (0.5 - v_threshold):
        directions.append("UP")
    elif avg_v_ratio > (0.5 + v_threshold):
        directions.append("DOWN")

    if not directions:
        return "CENTER"

    return " ".join(directions)


def calculate_iris_gaze_vector(face_landmarks, iris_indices, eye_center_indices, w, h):
    """Calculate a 3D gaze vector based on iris displacement from eye center."""
    iris_center = np.mean([[face_landmarks[idx].x * w,
                            face_landmarks[idx].y * h,
                            face_landmarks[idx].z * w] for idx in iris_indices], axis=0)

    eye_center = np.mean([[face_landmarks[idx].x * w,
                           face_landmarks[idx].y * h,
                           face_landmarks[idx].z * w] for idx in eye_center_indices], axis=0)

    gaze_vector = iris_center - eye_center

    norm = np.linalg.norm(gaze_vector)
    if norm > 1e-6:
        gaze_vector = gaze_vector / norm

    return gaze_vector


class SimpleKalmanFilter:
    """
    Simple 2D Kalman filter for iris position smoothing.
    Used when filterpy is not available.
    """
    def __init__(self, process_noise=0.01, measurement_noise=0.1):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.state = np.array([0.5, 0.5])  # [h_ratio, v_ratio]
        self.covariance = np.eye(2) * 0.1
    
    def update(self, measurement):
        """Update filter with new measurement"""
        # Predict
        Q = np.eye(2) * self.process_noise
        self.covariance = self.covariance + Q
        
        # Update
        R = np.eye(2) * self.measurement_noise
        K = self.covariance @ np.linalg.inv(self.covariance + R)
        self.state = self.state + K @ (measurement - self.state)
        self.covariance = (np.eye(2) - K) @ self.covariance
        
        return self.state.copy()
    
    def predict(self):
        """Get current state estimate"""
        return self.state.copy()


# =============================================================================
# AUTOMATIC GAZE CALIBRATION SYSTEM (NO USER INTERACTION REQUIRED)
# =============================================================================

class ScreenConfiguration:
    """
    Screen and camera configuration with reasonable defaults.
    No explicit user calibration needed - uses common setup assumptions.
    """
    def __init__(self, 
                 screen_width_cm=34.0,      # ~13-14" laptop typical width
                 screen_height_cm=19.0,      # 16:9 aspect ratio
                 camera_to_screen_cm=1.0,    # Camera usually at top of screen
                 typical_viewing_distance_cm=50.0):  # ~20 inches
        
        self.screen_width_cm = screen_width_cm
        self.screen_height_cm = screen_height_cm
        self.camera_to_screen_cm = camera_to_screen_cm
        self.typical_viewing_distance_cm = typical_viewing_distance_cm
        
        # Screen resolution (pixels)
        self.screen_width_px = MONITOR_WIDTH
        self.screen_height_px = MONITOR_HEIGHT
        
        # Pixels per cm ratio
        self.px_per_cm_x = self.screen_width_px / self.screen_width_cm
        self.px_per_cm_y = self.screen_height_px / self.screen_height_cm
        
        print(f"[ScreenConfig] Screen: {screen_width_cm}x{screen_height_cm} cm, "
              f"Resolution: {self.screen_width_px}x{self.screen_height_px} px")
        print(f"[ScreenConfig] Typical viewing distance: {typical_viewing_distance_cm} cm")


class AutoGazeCalibrator:
    """
    Automatic gaze calibration that adapts WITHOUT explicit user calibration.
    Perfect for students with low cognition - just start using it!
    
    Key Features:
    - Passive observation during first 10-15 seconds
    - Continuous adaptive learning
    - No explicit calibration steps required
    - Assumes natural viewing behavior
    """
    def __init__(self, adaptation_rate=0.02, initial_observation_time=10.0):
        self.adaptation_rate = adaptation_rate  # How fast to adapt (0.01-0.05)
        self.initial_observation_time = initial_observation_time  # Seconds
        
        # Calibration state
        self.is_initial_observation_complete = False
        self.start_time = time.time()
        self.observation_samples = []
        
        # Adaptive offsets (learned automatically)
        self.gaze_offset_x = 0.0  # pixels
        self.gaze_offset_y = 0.0  # pixels
        
        # Bias corrections (iris tends to appear more in certain positions)
        self.iris_h_bias = 0.0  # Horizontal bias correction
        self.iris_v_bias = 0.0  # Vertical bias correction
        
        # Running statistics for adaptive calibration
        self.center_gaze_samples = []  # Samples when looking near center
        self.max_center_samples = 100
        
        print(f"[AutoCalibrator] Passive observation mode: {initial_observation_time}s")
        print(f"[AutoCalibrator] No user action needed - just use naturally!")
    
    def add_observation(self, gaze_screen_x, gaze_screen_y, head_yaw, head_pitch):
        """
        Passively collect observations during initial period.
        Assumes user naturally looks around the screen.
        """
        current_time = time.time() - self.start_time
        
        if not self.is_initial_observation_complete:
            if current_time < self.initial_observation_time:
                self.observation_samples.append({
                    'x': gaze_screen_x,
                    'y': gaze_screen_y,
                    'yaw': head_yaw,
                    'pitch': head_pitch,
                    'time': current_time
                })
            else:
                self._complete_initial_observation()
    
    def _complete_initial_observation(self):
        """
        Process initial observations to set baseline calibration.
        Assumption: User looked around screen naturally during observation.
        """
        if len(self.observation_samples) < 10:
            print("[AutoCalibrator] Warning: Few samples collected, using defaults")
            self.is_initial_observation_complete = True
            return
        
        # Calculate center tendency (most people look at center most of time)
        gaze_x_values = [s['x'] for s in self.observation_samples]
        gaze_y_values = [s['y'] for s in self.observation_samples]
        
        # Use median instead of mean (more robust to outliers)
        median_x = np.median(gaze_x_values)
        median_y = np.median(gaze_y_values)
        
        screen_center_x = MONITOR_WIDTH / 2
        screen_center_y = MONITOR_HEIGHT / 2
        
        # Initial offset: difference between estimated gaze center and screen center
        self.gaze_offset_x = screen_center_x - median_x
        self.gaze_offset_y = screen_center_y - median_y
        
        print(f"[AutoCalibrator] Initial observation complete!")
        print(f"[AutoCalibrator] Auto-detected offset: ({self.gaze_offset_x:.1f}, {self.gaze_offset_y:.1f}) px")
        
        self.is_initial_observation_complete = True
        self.observation_samples.clear()  # Free memory
    
    def adaptive_update(self, raw_gaze_x, raw_gaze_y, head_yaw, head_pitch):
        """
        Continuously adapt calibration based on natural use patterns.
        
        Key insight: When head is neutral and eyes are centered in sockets,
        user is likely looking at screen center or current focus area.
        """
        # Only adapt after initial observation
        if not self.is_initial_observation_complete:
            return raw_gaze_x, raw_gaze_y
        
        # Apply current calibration
        calibrated_x = raw_gaze_x + self.gaze_offset_x
        calibrated_y = raw_gaze_y + self.gaze_offset_y
        
        # Adaptive learning: If head is relatively neutral, assume looking at center region
        # This is a gentle, continuous adaptation
        head_is_neutral = (abs(head_yaw) < 8.0 and abs(head_pitch) < 8.0)
        
        if head_is_neutral:
            screen_center_x = MONITOR_WIDTH / 2
            screen_center_y = MONITOR_HEIGHT / 2
            
            # Gentle drift correction toward center assumption
            error_x = screen_center_x - calibrated_x
            error_y = screen_center_y - calibrated_y
            
            # Only apply very small corrections (adaptive learning)
            self.gaze_offset_x += error_x * self.adaptation_rate
            self.gaze_offset_y += error_y * self.adaptation_rate
            
            # Recalculate with updated offset
            calibrated_x = raw_gaze_x + self.gaze_offset_x
            calibrated_y = raw_gaze_y + self.gaze_offset_y
        
        # Clamp to screen bounds
        calibrated_x = np.clip(calibrated_x, 0, MONITOR_WIDTH)
        calibrated_y = np.clip(calibrated_y, 0, MONITOR_HEIGHT)
        
        return calibrated_x, calibrated_y
    
    def apply_calibration(self, raw_gaze_x, raw_gaze_y):
        """
        Simple calibration application (no adaptation).
        Use this when you don't want adaptive updates.
        """
        if not self.is_initial_observation_complete:
            return raw_gaze_x, raw_gaze_y
        
        calibrated_x = raw_gaze_x + self.gaze_offset_x
        calibrated_y = raw_gaze_y + self.gaze_offset_y
        
        calibrated_x = np.clip(calibrated_x, 0, MONITOR_WIDTH)
        calibrated_y = np.clip(calibrated_y, 0, MONITOR_HEIGHT)
        
        return calibrated_x, calibrated_y
    
    def get_status(self):
        """Get calibration status for display"""
        if not self.is_initial_observation_complete:
            elapsed = time.time() - self.start_time
            remaining = max(0, self.initial_observation_time - elapsed)
            return f"Learning... {remaining:.0f}s"
        else:
            return "Active"


def calculate_3d_gaze_vector_with_head_pose(face_landmarks, iris_indices, eye_center_indices, 
                                             head_yaw, head_pitch, head_roll, w, h):
    """
    Calculate 3D gaze vector corrected for head pose.
    This is more accurate than iris position alone.
    
    Args:
        face_landmarks: MediaPipe face landmarks
        iris_indices: Iris landmark indices
        eye_center_indices: Eye center landmark indices
        head_yaw, head_pitch, head_roll: Head pose angles in degrees
        w, h: Frame dimensions
    
    Returns:
        3D gaze vector in camera coordinate system (normalized)
    """
    # Get iris-based gaze vector (in face coordinate system)
    iris_gaze = calculate_iris_gaze_vector(face_landmarks, iris_indices, 
                                           eye_center_indices, w, h)
    
    # Convert head pose angles to rotation matrix
    # This tells us head orientation relative to camera
    yaw_rad = np.radians(head_yaw)
    pitch_rad = np.radians(head_pitch)
    roll_rad = np.radians(head_roll)
    
    # Create rotation matrix from Euler angles (ZYX order)
    Rz = np.array([
        [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
        [np.sin(yaw_rad), np.cos(yaw_rad), 0],
        [0, 0, 1]
    ])
    
    Ry = np.array([
        [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
        [0, 1, 0],
        [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]
    ])
    
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll_rad), -np.sin(roll_rad)],
        [0, np.sin(roll_rad), np.cos(roll_rad)]
    ])
    
    # Combined rotation: head pose in camera space
    R_head = Rz @ Ry @ Rx
    
    # Transform iris gaze vector from face space to camera space
    gaze_vector_camera = R_head @ iris_gaze
    
    # Normalize
    norm = np.linalg.norm(gaze_vector_camera)
    if norm > 1e-6:
        gaze_vector_camera = gaze_vector_camera / norm
    
    return gaze_vector_camera


def gaze_vector_to_screen_coordinates(gaze_vector_3d, face_center_3d, 
                                      screen_config, estimated_distance_cm=None):
    """
    Project 3D gaze vector to 2D screen coordinates.
    
    Args:
        gaze_vector_3d: 3D gaze direction vector (normalized)
        face_center_3d: 3D position of face center in camera space
        screen_config: ScreenConfiguration object
        estimated_distance_cm: Estimated distance to screen (or None to use typical)
    
    Returns:
        (screen_x, screen_y) in pixels
    """
    # Use typical viewing distance if not provided
    if estimated_distance_cm is None:
        distance_cm = screen_config.typical_viewing_distance_cm
    else:
        distance_cm = estimated_distance_cm
    
    # Screen plane is at distance_cm from camera, aligned with camera
    # Camera is at top center of screen
    # Screen coordinates: origin at top-left
    
    # Calculate where gaze ray intersects screen plane
    # Gaze ray: P = face_center + t * gaze_vector
    # Screen plane: Z = distance_cm
    
    # Assuming face_center_3d[2] represents depth from camera
    # We need to find t such that: face_center_3d[2] + t * gaze_vector_3d[2] = distance_cm
    
    if abs(gaze_vector_3d[2]) < 1e-6:
        # Gaze vector parallel to screen - use center
        return MONITOR_WIDTH / 2, MONITOR_HEIGHT / 2
    
    # Calculate intersection parameter
    t = (distance_cm - face_center_3d[2]) / gaze_vector_3d[2]
    
    if t < 0:
        # Looking away from screen
        return MONITOR_WIDTH / 2, MONITOR_HEIGHT / 2
    
    # Intersection point in 3D
    intersection_x = face_center_3d[0] + t * gaze_vector_3d[0]
    intersection_y = face_center_3d[1] + t * gaze_vector_3d[1]
    
    # Convert to screen coordinates
    # Camera is at (screen_width/2, camera_to_screen_cm) relative to screen top-left
    # X: positive right, Y: positive down
    
    screen_center_x = screen_config.screen_width_cm / 2
    screen_center_y = screen_config.camera_to_screen_cm
    
    # Offset from screen center (in cm)
    offset_x_cm = intersection_x - screen_center_x
    offset_y_cm = intersection_y - screen_center_y
    
    # Convert to pixels
    screen_x = (screen_config.screen_width_cm / 2 + offset_x_cm) * screen_config.px_per_cm_x
    screen_y = offset_y_cm * screen_config.px_per_cm_y
    
    # Ensure within screen bounds
    screen_x = np.clip(screen_x, 0, MONITOR_WIDTH)
    screen_y = np.clip(screen_y, 0, MONITOR_HEIGHT)
    
    return float(screen_x), float(screen_y)


def estimate_face_distance_cm(face_landmarks, w, h, typical_eye_distance_cm=6.3):
    """
    Estimate distance from camera to face using eye separation.
    
    Args:
        face_landmarks: MediaPipe face landmarks
        w, h: Frame dimensions
        typical_eye_distance_cm: Average human interpupillary distance (IPD)
    
    Returns:
        Estimated distance in centimeters
    """
    # Get eye corner landmarks (outer corners)
    left_eye_outer = face_landmarks[33]  # Left eye left corner
    right_eye_outer = face_landmarks[263]  # Right eye right corner
    
    # Calculate eye separation in pixels
    eye_sep_x = abs(right_eye_outer.x - left_eye_outer.x) * w
    eye_sep_y = abs(right_eye_outer.y - left_eye_outer.y) * h
    eye_sep_px = np.sqrt(eye_sep_x**2 + eye_sep_y**2)
    
    if eye_sep_px < 1:
        return 50.0  # Default fallback
    
    # Estimate distance using similar triangles
    # Assuming typical webcam FOV ~60-70 degrees
    # Simplified: distance ≈ (eye_distance_real * frame_width) / (eye_distance_pixels * 2 * tan(FOV/2))
    
    # Rough estimate using proportionality
    # If eyes are X pixels apart, and typical is ~6.3cm, estimate distance
    typical_fov_horizontal = 65  # degrees (typical webcam)
    sensor_width_estimate = 0.5  # cm (typical webcam sensor)
    
    # Simplified distance estimation
    distance_cm = (typical_eye_distance_cm * w) / (eye_sep_px * 2 * np.tan(np.radians(typical_fov_horizontal / 2)))
    
    # Clamp to reasonable range (20cm to 100cm)
    distance_cm = np.clip(distance_cm, 20.0, 100.0)
    
    return float(distance_cm)


def detect_iris_opencv(eye_roi_image, eye_roi_info):
    """
    Detect iris center using OpenCV methods (Hough circles + contour detection).
    This is part of Option 3A hybrid approach.
    
    Args:
        eye_roi_image: Cropped eye region image (grayscale)
        eye_roi_info: ROI info dict from extract_eye_roi_info()
    
    Returns:
        (iris_x, iris_y) in normalized coordinates [0-1], or None if not detected
    """
    try:
        if eye_roi_image is None or eye_roi_image.size == 0:
            return None
        
        # Convert to grayscale if needed
        if len(eye_roi_image.shape) == 3:
            gray = cv2.cvtColor(eye_roi_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = eye_roi_image.copy()
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Method 1: Hough Circles for iris detection
        h, w = gray.shape
        min_radius = max(3, int(min(h, w) * 0.15))
        max_radius = max(5, int(min(h, w) * 0.4))
        
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=max_radius * 2,
            param1=50,
            param2=30,
            minRadius=min_radius,
            maxRadius=max_radius
        )
        
        iris_center = None
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            # Use the circle closest to center (most likely iris)
            center_x, center_y = w // 2, h // 2
            best_circle = None
            best_dist = float('inf')
            
            for (x, y, r) in circles:
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if dist < best_dist:
                    best_dist = dist
                    best_circle = (x, y)
            
            if best_circle:
                iris_center = best_circle
        
        # Method 2: Contour detection (fallback)
        if iris_center is None:
            # Use adaptive threshold
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Find contours
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            if contours:
                # Find largest contour (likely iris)
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Get bounding circle
                (x, y), radius = cv2.minEnclosingCircle(largest_contour)
                
                # Filter by size (iris should be reasonable size)
                if min_radius <= radius <= max_radius:
                    iris_center = (int(x), int(y))
        
        # Convert to normalized coordinates
        if iris_center and eye_roi_info:
            roi_x, roi_y, roi_w, roi_h = eye_roi_info['bbox']
            center_x, center_y = eye_roi_info['center']
            scale = eye_roi_info['scale']
            
            # Convert ROI coordinates to full frame coordinates
            full_x = roi_x + iris_center[0]
            full_y = roi_y + iris_center[1]
            
            # Normalize relative to ROI center
            norm_x = (full_x - center_x) / scale if scale > 0 else 0.0
            norm_y = (full_y - center_y) / scale if scale > 0 else 0.0
            
            return (norm_x, norm_y)
        
        return None
        
    except Exception as e:
        return None


def get_iris_position_ratio_hybrid(face_landmarks, iris_indices, eye_roi_info, 
                                   eye_roi_image, w, h, kalman_filter=None):
    """
    Hybrid approach: Combine MediaPipe landmarks with OpenCV detection.
    This is Option 3A implementation.
    
    Args:
        face_landmarks: MediaPipe face landmarks
        iris_indices: List of iris landmark indices
        eye_roi_info: ROI info dict
        eye_roi_image: Cropped eye ROI image
        w, h: Frame dimensions
        kalman_filter: Optional Kalman filter for smoothing
    
    Returns:
        (horizontal_ratio, vertical_ratio) tuple
    """
    try:
        # Method 1: MediaPipe landmarks (primary)
        mp_h_ratio, mp_v_ratio = get_iris_position_ratio_roi(
            face_landmarks, iris_indices, eye_roi_info, w, h
        )
        
        # Method 2: OpenCV detection (secondary)
        opencv_result = detect_iris_opencv(eye_roi_image, eye_roi_info)
        
        if opencv_result is not None:
            # Convert OpenCV normalized coords to ratios
            opencv_norm_x, opencv_norm_y = opencv_result
            
            # Get normalized eye boundary
            normalized_landmarks = eye_roi_info['normalized_landmarks']
            eye_left_norm = np.min(normalized_landmarks[:, 0])
            eye_right_norm = np.max(normalized_landmarks[:, 0])
            eye_top_norm = np.min(normalized_landmarks[:, 1])
            eye_bottom_norm = np.max(normalized_landmarks[:, 1])
            
            eye_width_norm = eye_right_norm - eye_left_norm
            eye_height_norm = eye_bottom_norm - eye_top_norm
            
            if eye_width_norm > 0:
                opencv_h_ratio = (opencv_norm_x - eye_left_norm) / eye_width_norm
            else:
                opencv_h_ratio = 0.5
            
            if eye_height_norm > 0:
                opencv_v_ratio = (opencv_norm_y - eye_top_norm) / eye_height_norm
            else:
                opencv_v_ratio = 0.5
            
            opencv_v_ratio = 1.0 - opencv_v_ratio  # Invert
            
            # Weighted combination: 70% MediaPipe, 30% OpenCV
            # MediaPipe is more reliable, OpenCV helps with edge cases
            weight_mp = 0.7
            weight_cv = 0.3
            
            combined_h = weight_mp * mp_h_ratio + weight_cv * opencv_h_ratio
            combined_v = weight_mp * mp_v_ratio + weight_cv * opencv_v_ratio
        else:
            # Fallback to MediaPipe only
            combined_h, combined_v = mp_h_ratio, mp_v_ratio
        
        # Apply Kalman filtering if available
        if kalman_filter is not None:
            measurement = np.array([combined_h, combined_v])
            # Handle both filterpy KalmanFilter and SimpleKalmanFilter
            if hasattr(kalman_filter, 'update'):
                if KALMAN_AVAILABLE and hasattr(kalman_filter, 'predict'):
                    # filterpy KalmanFilter
                    kalman_filter.predict()
                    kalman_filter.update(measurement)
                    filtered = kalman_filter.x[:2]  # Get first 2 elements
                else:
                    # SimpleKalmanFilter
                    filtered = kalman_filter.update(measurement)
                combined_h, combined_v = float(filtered[0]), float(filtered[1])
        
        combined_h = np.clip(combined_h, 0.0, 1.0)
        combined_v = np.clip(combined_v, 0.0, 1.0)
        
        return combined_h, combined_v
        
    except Exception:
        return 0.5, 0.5


def smooth_eye_ratios(ratio_buffer, new_h_ratio, new_v_ratio):
    """
    Smooth eye ratios using exponential moving average.
    
    Args:
        ratio_buffer: deque buffer containing (h_ratio, v_ratio) tuples
        new_h_ratio: New horizontal ratio
        new_v_ratio: New vertical ratio
    
    Returns:
        (smoothed_h_ratio, smoothed_v_ratio) tuple
    """
    ratio_buffer.append((new_h_ratio, new_v_ratio))
    
    if len(ratio_buffer) == 0:
        return new_h_ratio, new_v_ratio
    
    # Use exponential moving average with more weight on recent values
    weights = np.exp(np.linspace(-1, 0, len(ratio_buffer)))
    weights = weights / weights.sum()
    
    h_ratios = np.array([r[0] for r in ratio_buffer])
    v_ratios = np.array([r[1] for r in ratio_buffer])
    
    smoothed_h = np.average(h_ratios, weights=weights)
    smoothed_v = np.average(v_ratios, weights=weights)
    
    return float(smoothed_h), float(smoothed_v)


def smooth_gaze_direction(new_direction, direction_buffer, buffer_size=5):
    """Smooth gaze direction detection using a voting system."""
    direction_buffer.append(new_direction)
    if len(direction_buffer) > buffer_size:
        direction_buffer.pop(0)

    direction_counts = {}
    for direction in direction_buffer:
        if direction not in direction_counts:
            direction_counts[direction] = 0
        direction_counts[direction] += 1

    if direction_counts:
        return max(direction_counts, key=direction_counts.get)
    return "CENTER"

def rotation_matrix_to_euler_angles(R):
    """
    Calculates Yaw, Pitch, Roll from a 3x3 rotation matrix.
    Also handles the gimbal lock problem.
    Based on: https://stackoverflow.com/questions/43364900/rotation-matrix-to-euler-angles-with-opencv
    
    """
    
    # Check for singularity at north pole
    if R[1, 0] > 0.998:  
        yaw = math.atan2(R[0, 2], R[2, 2])
        pitch = math.pi / 2
        roll = 0
        return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)
    
    # Check for singularity at south pole
    if R[1, 0] < -0.998: 
        yaw = math.atan2(R[0, 2], R[2, 2])
        pitch = -math.pi / 2
        roll = 0
        return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)

    # General case
    yaw = math.atan2(-R[2, 0], R[0, 0])
    pitch = math.asin(R[1, 0])
    roll = math.atan2(-R[1, 2], R[1, 1])
    
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


def get_head_pose(face_landmarks, w, h):
    """
    Calculates head pose (Yaw, Pitch, Roll) using cv2.solvePnP.
    This is the "Pro Fix" method.
    """
    
    # 1. Define 3D Canonical Face Model Points
    #
    # These 6 points are: Nose tip, Chin, Left eye corner, Right eye corner,
    # Left mouth corner, Right mouth corner
    model_points_3d = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left Mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ], dtype=np.float64)

    # 2. Get 2D Image Points from MediaPipe
    # MediaPipe landmark indices for the 6 points
    # (Nose: 1, Chin: 199, L-Eye: 33, R-Eye: 263, L-Mouth: 61, R-Mouth: 291)
    image_points_2d = np.array([
        (face_landmarks[1].x * w, face_landmarks[1].y * h),    # Nose tip
        (face_landmarks[199].x * w, face_landmarks[199].y * h), # Chin
        (face_landmarks[33].x * w, face_landmarks[33].y * h),  # Left eye corner
        (face_landmarks[263].x * w, face_landmarks[263].y * h), # Right eye corner
        (face_landmarks[61].x * w, face_landmarks[61].y * h),  # Left mouth corner
        (face_landmarks[291].x * w, face_landmarks[291].y * h)   # Right mouth corner
    ], dtype=np.float64)

    # 3. Estimate Camera Matrix
    #
    focal_length = w
    center_x = w / 2
    center_y = h / 2
    camera_matrix = np.array([
        [focal_length, 0, center_x],
        [0, focal_length, center_y],
        [0, 0, 1]
    ], dtype=np.float64)

    # 4. Assume No Lens Distortion
    #
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    # 5. Call cv2.solvePnP
    #
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points_3d, 
        image_points_2d, 
        camera_matrix, 
        dist_coeffs, 
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return 0.0, 0.0, 0.0 # Return neutral on failure

    # 6. Get Rotation Matrix from vector
    #
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # 7. Decompose Rotation Matrix into Euler Angles
    yaw, pitch, roll = rotation_matrix_to_euler_angles(rotation_matrix)
    
    # The solvePnP axes are different. We need to adjust.
    # This is a common adjustment step found through trial-and-error
    # based on the 3D model's orientation.
    # Yaw (Y-axis): Needs to be inverted
    # Pitch (X-axis): Good
    # Roll (Z-axis): Good
    
    # We return degrees
    return -yaw, pitch, roll

def get_gaze_direction(yaw, pitch, threshold=0.1):
    """Determine gaze direction based on yaw and pitch angles"""
    # Threshold in radians for determining direction
    # threshold of 0.1 radians ≈ 5.7 degrees
    
    directions = []
    
    # Horizontal direction (yaw)
    if yaw < -threshold:
        directions.append("LEFT")
    elif yaw > threshold:
        directions.append("RIGHT")
    
    # Vertical direction (pitch)
    if pitch < -threshold:
        directions.append("DOWN")
    elif pitch > threshold:
        directions.append("UP")
    
    # If no significant direction, return center
    if not directions:
        return "CENTER"
    
    return " ".join(directions)

def get_head_direction(yaw_deg, pitch_deg, threshold=10.0):
    """
    Determine head direction (string) from yaw/pitch in DEGREES.
    Used for the raw visual indicator.
    """
    directions = []
    
    if yaw_deg < -threshold:
        directions.append("LEFT")
    elif yaw_deg > threshold:
        directions.append("RIGHT")
    
    if pitch_deg < -threshold:
        directions.append("DOWN")
    elif pitch_deg > threshold:
        directions.append("UP")
    
    if not directions:
        return "CENTER"
    
    return " ".join(directions)

def draw_wireframe_cube(frame, center, R, size=80):
    """Draw a 3D wireframe cube showing head orientation"""
    # Extract axis directions from rotation matrix
    right = R[:, 0]
    up = -R[:, 1]
    forward = -R[:, 2]
    
    # Half-sizes
    hw, hh, hd = size * 1, size * 1, size * 1
    
    def corner(x_sign, y_sign, z_sign):
        """Calculate corner position"""
        return (center + 
                x_sign * hw * right + 
                y_sign * hh * up + 
                z_sign * hd * forward)
    
    # 8 corners of the cube
    corners = [corner(x, y, z) for x in [-1, 1] for y in [1, -1] for z in [-1, 1]]
    projected = [(int(pt[0]), int(pt[1])) for pt in corners]
    
    # Define edges connecting corners
    edges = [
        (0, 1), (1, 3), (3, 2), (2, 0),  # Front face
        (4, 5), (5, 7), (7, 6), (6, 4),  # Back face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting edges
    ]
    
    # Draw cube edges
    for i, j in edges:
        cv2.line(frame, projected[i], projected[j], (255, 128, 0), 2)

def draw_coordinate_axes(frame, center, R, length=100):
    """Draw X, Y, Z coordinate axes showing head orientation"""
    # R = [right, up, -forward]
    # Extract axis directions
    x_axis = R[:, 0] * length      # Right direction
    y_axis = R[:, 1] * length      # Up direction (already correct in R)
    z_axis = -R[:, 2] * length     # Forward direction
    
    center_2d = (int(center[0]), int(center[1]))
    
    # X-axis: Red (right direction)
    x_end = (int(center[0] + x_axis[0]), int(center[1] + x_axis[1]))
    cv2.arrowedLine(frame, center_2d, x_end, (0, 0, 255), 3, tipLength=0.2)
    cv2.putText(frame, "R", x_end, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Y-axis: Green (up direction)
    y_end = (int(center[0] + y_axis[0]), int(center[1] + y_axis[1]))
    cv2.arrowedLine(frame, center_2d, y_end, (0, 255, 0), 3, tipLength=0.2)
    cv2.putText(frame, "U", y_end, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Z-axis: Blue (forward direction - where you're looking)
    z_end = (int(center[0] + z_axis[0]), int(center[1] + z_axis[1]))
    cv2.arrowedLine(frame, center_2d, z_end, (255, 0, 0), 3, tipLength=0.2)
    cv2.putText(frame, "F", z_end, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

def draw_direction_arrow(frame, direction, center_x, center_y, length=40, color=(255, 255, 255)):
    """Draw an arrow indicating direction"""
    # Calculate arrow endpoints based on direction
    if "LEFT" in direction and "UP" in direction:
        dx, dy = -length, -length
    elif "LEFT" in direction and "DOWN" in direction:
        dx, dy = -length, length
    elif "RIGHT" in direction and "UP" in direction:
        dx, dy = length, -length
    elif "RIGHT" in direction and "DOWN" in direction:
        dx, dy = length, length
    elif "LEFT" in direction:
        dx, dy = -length, 0
    elif "RIGHT" in direction:
        dx, dy = length, 0
    elif "UP" in direction:
        dx, dy = 0, -length
    elif "DOWN" in direction:
        dx, dy = 0, length
    else:  # CENTER
        # Draw a circle for center
        cv2.circle(frame, (center_x, center_y), 8, color, 2)
        return
    
    # Draw arrow line
    end_x, end_y = center_x + int(dx), center_y + int(dy)
    cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y), color, 3, tipLength=0.3)

def draw_eye_metrics(frame, ear=0, gaze_direction=None, head_direction=None, 
                     head_yaw=0, head_pitch=0, head_roll=0, 
                     student_gesture_msg="", gesture_state="", 
                     looking_up=False, eye_ratio=0.5,
                     mouth_open=False, mouth_ratio=0.0):
    """Draw eye tracking metrics on frame"""
    global last_time, fps_counter
    
    h, w = frame.shape[:2]
    y_offset = 30
    
    # FPS
    current_time = time.time()
    fps = 1.0 / (current_time - last_time + 0.0001)
    fps_counter.append(fps)
    avg_fps = np.mean(fps_counter)
    last_time = current_time
    
    cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    y_offset += 25
    
    # MOUTH OPEN detection (NEW - no calibration needed!)
    mouth_color = (0, 255, 0) if mouth_open else (100, 100, 100)
    mouth_text = "*** MOUTH OPEN! ***" if mouth_open else "Mouth: Closed"
    
    # Make it extra visible when mouth is open
    font_scale = 1.2 if mouth_open else 0.6
    thickness = 3 if mouth_open else 2
    
    cv2.putText(frame, f"{mouth_text}", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, mouth_color, thickness)
    y_offset += 35 if mouth_open else 25
    
    # Always show the mouth ratio for debugging
    cv2.putText(frame, f"Mouth Ratio: {mouth_ratio:.4f} (>0.50=OPEN)", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    y_offset += 25
    
    # Head direction (LEFT/RIGHT/UP/DOWN) with visual indicator
    # This now displays the *raw* head direction
    if head_direction:
        # Use different colors for different directions
        if "LEFT" in head_direction:
            head_color = (0, 165, 255)  # Orange
        elif "RIGHT" in head_direction:
            head_color = (255, 0, 255)  # Magenta
        elif "UP" in head_direction:
            head_color = (255, 255, 0)  # Cyan
        elif "DOWN" in head_direction:
            head_color = (0, 255, 255)  # Yellow
        else:
            head_color = (0, 255, 0)  # Green for center
        
        # Draw text with angles (degrees)
        # Note: head_yaw, head_pitch, head_roll are already in degrees
        head_text = f"Head: {head_direction} Y:{head_yaw:.0f} P:{head_pitch:.0f} R:{head_roll:.0f}"
        cv2.putText(frame, head_text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, head_color, 2)
        
        # Draw directional arrow indicator in top right
        arrow_center_x = w - 60
        arrow_center_y = 60
        cv2.circle(frame, (arrow_center_x, arrow_center_y), 50, (100, 100, 100), 2)
        draw_direction_arrow(frame, head_direction, arrow_center_x, arrow_center_y, 35, head_color)
        
        # Draw roll indicator (horizontal line that tilts)
        roll_indicator_length = 40
        roll_x1 = int(arrow_center_x - roll_indicator_length * math.cos(head_roll * math.pi / 180))
        roll_y1 = int(arrow_center_y - roll_indicator_length * math.sin(head_roll * math.pi / 180))
        roll_x2 = int(arrow_center_x + roll_indicator_length * math.cos(head_roll * math.pi / 180))
        roll_y2 = int(arrow_center_y + roll_indicator_length * math.sin(head_roll * math.pi / 180))
        cv2.line(frame, (roll_x1, roll_y1), (roll_x2, roll_y2), (255, 255, 255), 2)
        
        y_offset += 30
    
    # Gaze direction (LEFT/RIGHT/UP/DOWN)
    if gaze_direction:
        # Use different colors for different directions
        if "LEFT" in gaze_direction:
            dir_color = (0, 165, 255)  # Orange
        elif "RIGHT" in gaze_direction:
            dir_color = (255, 0, 255)  # Magenta
        elif "UP" in gaze_direction:
            dir_color = (255, 255, 0)  # Cyan
        elif "DOWN" in gaze_direction:
            dir_color = (0, 255, 255)  # Yellow
        else:
            dir_color = (0, 255, 0)  # Green for center
        
        cv2.putText(frame, f"Gaze: {gaze_direction}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, dir_color, 2)
        y_offset += 30
    
    # Blink counter
    cv2.putText(frame, f"Blinks: {total_blinks}", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    y_offset += 25
    
    # Eye aspect ratio
    if ear > 0:
        cv2.putText(frame, f"EAR: {ear:.3f}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 25
    
    # Mouse control status
    status_color = (0, 255, 0) if mouse_control_enabled else (0, 0, 255)
    cv2.putText(frame, f"Mouse: {'ON' if mouse_control_enabled else 'OFF'}", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    y_offset += 25
    
    # === NEW: Student Gesture System Status ===
    cv2.putText(frame, f"Gesture State: {gesture_state}", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    y_offset += 25
    
    if student_gesture_msg:
        # Use a prominent color for detected gestures
        gesture_color = (0, 255, 255) # Yellow
        if "QUICK" in student_gesture_msg or "SLOW" in student_gesture_msg:
            gesture_color = (0, 255, 0) # Bright Green
        elif "CALIBRATING" in student_gesture_msg:
            gesture_color = (0, 165, 255) # Orange
        elif "Failed" in student_gesture_msg:
            gesture_color = (0, 0, 255) # Red
            
        cv2.putText(frame, f"Gesture: {student_gesture_msg}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, gesture_color, 2)
        y_offset += 30 # Extra space
    # ==========================================
    
    # Calibration status
    if left_sphere_locked and right_sphere_locked:
        cv2.putText(frame, "Eye Calib: OK", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Press 'c' for EYE calib", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    y_offset += 25
    # Show head calib status
    if neutral_yaw != 0.0 or neutral_pitch != 0.0:
        cv2.putText(frame, "Head Calib: OK", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Press 'k' for HEAD calib", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)


def save_calibration(filename="calibration.json"):
    """Save calibration data to file"""
    config_path = get_config_path()
    filepath = config_path / filename
    
    data = {
        'calibration_offset_yaw': calibration_offset_yaw,
        'calibration_offset_pitch': calibration_offset_pitch,
        'left_sphere_locked': left_sphere_locked,
        'right_sphere_locked': right_sphere_locked,
        'monitor_size': (MONITOR_WIDTH, MONITOR_HEIGHT),
        'timestamp': datetime.now().isoformat()
    }
    
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"Calibration saved to {filepath}")
        return True
    except Exception as e:
        print(f"Failed to save calibration: {e}")
        return False

def load_calibration(filename="calibration.json"):
    """Load calibration data from file"""
    global calibration_offset_yaw, calibration_offset_pitch
    
    config_path = get_config_path()
    filepath = config_path / filename
    
    if filepath.exists():
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            calibration_offset_yaw = data.get('calibration_offset_yaw', 0)
            calibration_offset_pitch = data.get('calibration_offset_pitch', 0)
            print(f"Calibration loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Failed to load calibration: {e}")
    return False

# Main loop
def main():
    global mouse_control_enabled, high_contrast_mode, enable_logging, show_cube
    global left_sphere_locked, right_sphere_locked, left_sphere_local_offset, right_sphere_local_offset
    global left_calibration_nose_scale, right_calibration_nose_scale
    global monitor_corners, monitor_center_w, monitor_normal_w, units_per_cm
    global debug_world_frozen, orbit_pivot_frozen
    global calibration_offset_yaw, calibration_offset_pitch
    
    # --- NEW: Gesture state globals ---
    global gesture_state, gesture_start_time, gesture_direction, neutral_yaw, neutral_pitch
    global calibration_start_time, calibration_readings_yaw, calibration_readings_pitch
    global last_gesture_time, last_gesture_detected, gesture_persistence_counter
    global gesture_timeout_counter, GESTURE_TIMEOUT_THRESHOLD
    global last_mouth_state, last_looking_up_state, last_gaze_direction_spoken
    global left_eye_ratio_buffer, right_eye_ratio_buffer, use_roi_processing
    global use_hybrid_processing, left_eye_kalman, right_eye_kalman
    global detect_looking_up_enabled, optikey_enabled
    
    gaze_direction_buffer = []
    iris_calibration_h_offset = 0.0
    iris_calibration_v_offset = 0.0
    
    # Debug frame counter for gaze tracking
    gaze_debug_frame_count = 0
    
    # Initialize ROI-based processing buffers
    if use_roi_processing:
        left_eye_ratio_buffer = deque(maxlen=5)
        right_eye_ratio_buffer = deque(maxlen=5)
    
    # Initialize Kalman filters for hybrid approach
    if use_hybrid_processing:
        if KALMAN_AVAILABLE:
            from filterpy.kalman import KalmanFilter
            # Create 2D Kalman filters for each eye
            left_eye_kalman = KalmanFilter(dim_x=2, dim_z=2)
            left_eye_kalman.x = np.array([0.5, 0.5])  # Initial state
            left_eye_kalman.F = np.eye(2)  # State transition
            left_eye_kalman.H = np.eye(2)  # Measurement function
            left_eye_kalman.P *= 0.1  # Covariance
            left_eye_kalman.R *= 0.1  # Measurement noise
            left_eye_kalman.Q *= 0.01  # Process noise
            
            right_eye_kalman = KalmanFilter(dim_x=2, dim_z=2)
            right_eye_kalman.x = np.array([0.5, 0.5])
            right_eye_kalman.F = np.eye(2)
            right_eye_kalman.H = np.eye(2)
            right_eye_kalman.P *= 0.1
            right_eye_kalman.R *= 0.1
            right_eye_kalman.Q *= 0.01
        else:
            # Use simple Kalman filter
            left_eye_kalman = SimpleKalmanFilter()
            right_eye_kalman = SimpleKalmanFilter()
            print("[Hybrid Mode] Using simple Kalman filter (install filterpy for advanced filtering)")

    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open camera!")
        return
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Camera opened: {w}x{h}")
    
    # Initialize automatic gaze calibration system (NO USER CALIBRATION REQUIRED!)
    screen_config = ScreenConfiguration(
        screen_width_cm=34.0,       # Adjust for your screen (typical 13-14" laptop)
        screen_height_cm=19.0,      # 16:9 aspect ratio
        camera_to_screen_cm=1.0,    # Camera at top of screen
        typical_viewing_distance_cm=50.0  # ~20 inches typical
    )
    
    auto_calibrator = AutoGazeCalibrator(
        adaptation_rate=0.02,        # How fast to adapt
        initial_observation_time=10.0  # 10 seconds passive observation
    )
    
    # Flag to enable the new automatic gaze system
    use_auto_gaze = True
    
    # Load saved calibration if available
    load_calibration()
    
    print("\n" + "="*60)
    print("=== Enhanced Eye Tracker Controls ===")
    print("="*60)
    print("c - Calibrate EYE spheres (look at center)")
    print("k - Calibrate ADAPTIVE HEAD GESTURE system (hold neutral pose)")
    print("g - Calibrate iris gaze (look straight ahead)")
    print("t - Toggle ROI-based eye processing (currently: " + ("ON" if use_roi_processing else "OFF") + ")")
    print("y - Toggle Hybrid processing (MediaPipe + OpenCV) (currently: " + ("ON" if use_hybrid_processing else "OFF") + ")")
    print("a - Toggle AUTO GAZE tracking (NO calibration!) (currently: " + ("ON" if use_auto_gaze else "OFF") + ")")
    print("s - Save calibration")
    print("l - Load calibration")
    print("F7 - Toggle mouse control")
    print("h - Toggle high contrast mode")
    print("v - Toggle 3D cube/axes visualization")
    print("u - Toggle looking up detection (currently: " + ("ON" if detect_looking_up_enabled else "OFF") + ")")
    print("x - Place gaze marker")
    print("j/l - Orbit yaw left/right")
    print("i/k - Orbit pitch up/down")
    print("[ / ] - Zoom out/in")
    print("r - Reset view")
    print("q - Quit")
    print("="*60)
    print("\n=== Command Prompt Output ===")
    print("The following events will be printed here:")
    print("  - >>> MOUTH OPENED/CLOSED (with timestamps)")
    print("  - >>> LOOKING UP/CENTER (press 'u' to enable, ignored during blinks)")
    print("  - >>> GESTURE STARTED/COMPLETED/TIMEOUT")
    print("  - >>> HEAD GESTURE CALIBRATION")
    print("="*60)
    print("\n=== AUTO GAZE TRACKING (ENABLED BY DEFAULT) ===")
    print("Automatic eye gaze calibration - NO user action needed!")
    print("  - Grey 'Gaze Direction Visualization' window will open")
    print("  - Red dot shows where you're looking on screen")
    print("  - Mouse cursor will follow your gaze smoothly")
    print("  - System learns passively for first 10 seconds")
    print("  - Press 'a' to toggle AUTO GAZE on/off")
    print("="*60 + "\n")
    
    # Initialize head pose angles
    head_yaw, head_pitch, head_roll = 0.0, 0.0, 0.0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        # We need a raw head direction for the arrow indicator
        raw_head_direction = None

        smoothed_gaze_direction = "CENTER"
        left_h_ratio = right_h_ratio = left_v_ratio = right_v_ratio = 0.5
        left_iris_center = None
        right_iris_center = None
        gaze_overlay_direction = None
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark
            
            # --- START: "PRO FIX" HEAD POSE CALCULATION ---
            # This single function replaces the old PCA logic
            head_yaw, head_pitch, head_roll = get_head_pose(face_landmarks, w, h)
            # --- END: "PRO FIX" ---
            
            # === Iris-based gaze detection ===
            # OPTION 3A: Hybrid Processing (MediaPipe + OpenCV + Kalman)
            if use_hybrid_processing and use_roi_processing:
                # Extract eye ROI information
                left_eye_roi = extract_eye_roi_info(
                    face_landmarks, LEFT_EYE_LANDMARKS_71, w, h, padding_ratio=0.2
                )
                right_eye_roi = extract_eye_roi_info(
                    face_landmarks, RIGHT_EYE_LANDMARKS_71, w, h, padding_ratio=0.2
                )
                
                # Extract eye ROI images for OpenCV processing
                left_eye_image = None
                right_eye_image = None
                
                if left_eye_roi and right_eye_roi:
                    roi_x_l, roi_y_l, roi_w_l, roi_h_l = left_eye_roi['bbox']
                    roi_x_r, roi_y_r, roi_w_r, roi_h_r = right_eye_roi['bbox']
                    
                    # Crop eye regions from frame
                    left_eye_image = frame[roi_y_l:roi_y_l+roi_h_l, roi_x_l:roi_x_l+roi_w_l]
                    right_eye_image = frame[roi_y_r:roi_y_r+roi_h_r, roi_x_r:roi_x_r+roi_w_r]
                    
                    # Use hybrid method
                    left_h_ratio, left_v_ratio = get_iris_position_ratio_hybrid(
                        face_landmarks, left_iris_indices, left_eye_roi,
                        left_eye_image, w, h, left_eye_kalman
                    )
                    right_h_ratio, right_v_ratio = get_iris_position_ratio_hybrid(
                        face_landmarks, right_iris_indices, right_eye_roi,
                        right_eye_image, w, h, right_eye_kalman
                    )
                else:
                    # Fallback to ROI method
                    if left_eye_roi and right_eye_roi:
                        left_h_ratio, left_v_ratio = get_iris_position_ratio_roi(
                            face_landmarks, left_iris_indices, left_eye_roi, w, h
                        )
                        right_h_ratio, right_v_ratio = get_iris_position_ratio_roi(
                            face_landmarks, right_iris_indices, right_eye_roi, w, h
                        )
                    else:
                        left_h_ratio, left_v_ratio = get_iris_position_ratio(
                            face_landmarks, LEFT_EYE_CORNERS, LEFT_EYE_LIDS,
                            left_iris_indices, w, h,
                            eye_contour_indices=LEFT_EYE_LANDMARKS_71
                        )
                        right_h_ratio, right_v_ratio = get_iris_position_ratio(
                            face_landmarks, RIGHT_EYE_CORNERS, RIGHT_EYE_LIDS,
                            right_iris_indices, w, h,
                            eye_contour_indices=RIGHT_EYE_LANDMARKS_71
                        )
            
            # OPTION 2: ROI-Based Eye Processing for improved accuracy
            elif use_roi_processing:
                # Extract eye ROI information
                left_eye_roi = extract_eye_roi_info(
                    face_landmarks, LEFT_EYE_LANDMARKS_71, w, h, padding_ratio=0.2
                )
                right_eye_roi = extract_eye_roi_info(
                    face_landmarks, RIGHT_EYE_LANDMARKS_71, w, h, padding_ratio=0.2
                )
                
                # Calculate ratios using ROI-based method
                if left_eye_roi and right_eye_roi:
                    left_h_ratio, left_v_ratio = get_iris_position_ratio_roi(
                        face_landmarks, left_iris_indices, left_eye_roi, w, h
                    )
                    right_h_ratio, right_v_ratio = get_iris_position_ratio_roi(
                        face_landmarks, right_iris_indices, right_eye_roi, w, h
                    )
                    
                    # Apply independent smoothing to each eye
                    left_h_ratio, left_v_ratio = smooth_eye_ratios(
                        left_eye_ratio_buffer, left_h_ratio, left_v_ratio
                    )
                    right_h_ratio, right_v_ratio = smooth_eye_ratios(
                        right_eye_ratio_buffer, right_h_ratio, right_v_ratio
                    )
                else:
                    # Fallback to original method if ROI extraction fails
                    left_h_ratio, left_v_ratio = get_iris_position_ratio(
                        face_landmarks, LEFT_EYE_CORNERS, LEFT_EYE_LIDS,
                        left_iris_indices, w, h,
                        eye_contour_indices=LEFT_EYE_LANDMARKS_71
                    )
                    right_h_ratio, right_v_ratio = get_iris_position_ratio(
                        face_landmarks, RIGHT_EYE_CORNERS, RIGHT_EYE_LIDS,
                        right_iris_indices, w, h,
                        eye_contour_indices=RIGHT_EYE_LANDMARKS_71
                    )
            else:
                # Original method with 71 landmarks
                left_h_ratio, left_v_ratio = get_iris_position_ratio(
                    face_landmarks,
                    LEFT_EYE_CORNERS,
                    LEFT_EYE_LIDS,
                    left_iris_indices,
                    w, h,
                    eye_contour_indices=LEFT_EYE_LANDMARKS_71
                )

                right_h_ratio, right_v_ratio = get_iris_position_ratio(
                    face_landmarks,
                    RIGHT_EYE_CORNERS,
                    RIGHT_EYE_LIDS,
                    right_iris_indices,
                    w, h,
                    eye_contour_indices=RIGHT_EYE_LANDMARKS_71
                )

            # Apply calibration offsets
            left_h_ratio += iris_calibration_h_offset
            right_h_ratio += iris_calibration_h_offset
            left_v_ratio += iris_calibration_v_offset
            right_v_ratio += iris_calibration_v_offset

            left_h_ratio = float(np.clip(left_h_ratio, 0.0, 1.0))
            right_h_ratio = float(np.clip(right_h_ratio, 0.0, 1.0))
            left_v_ratio = float(np.clip(left_v_ratio, 0.0, 1.0))
            right_v_ratio = float(np.clip(right_v_ratio, 0.0, 1.0))

            raw_gaze_direction = get_combined_gaze_direction(
                left_h_ratio, left_v_ratio,
                right_h_ratio, right_v_ratio,
                h_threshold=0.25,
                v_threshold=0.2
            )

            smoothed_gaze_direction = smooth_gaze_direction(
                raw_gaze_direction,
                gaze_direction_buffer,
                buffer_size=7
            )

            left_gaze_vector = calculate_iris_gaze_vector(
                face_landmarks,
                left_iris_indices,
                left_eye_indices[:8],
                w, h
            )

            right_gaze_vector = calculate_iris_gaze_vector(
                face_landmarks,
                right_iris_indices,
                right_eye_indices[:8],
                w, h
            )

            _combined_gaze_vector = (left_gaze_vector + right_gaze_vector) / 2.0
            _combined_gaze_vector /= np.linalg.norm(_combined_gaze_vector) + 1e-9

            # ============================================================
            # NEW: AUTOMATIC GAZE POINT ESTIMATION (NO CALIBRATION!)
            # ============================================================
            if use_auto_gaze:
                # SIMPLIFIED METHOD: Use iris ratios directly to map to screen
                # This is more reliable than complex 3D projection
                
                # Average the iris position ratios from both eyes
                avg_h_ratio = (left_h_ratio + right_h_ratio) / 2.0
                avg_v_ratio = (left_v_ratio + right_v_ratio) / 2.0
                
                # Map iris ratios directly to screen coordinates
                # Apply head pose compensation
                head_yaw_factor = np.tan(np.radians(head_yaw)) * 0.3
                head_pitch_factor = np.tan(np.radians(head_pitch)) * 0.3
                
                # Adjust ratios based on head pose
                adjusted_h_ratio = avg_h_ratio + head_yaw_factor
                adjusted_v_ratio = avg_v_ratio + head_pitch_factor
                
                # Clamp to valid range
                adjusted_h_ratio = np.clip(adjusted_h_ratio, 0.0, 1.0)
                adjusted_v_ratio = np.clip(adjusted_v_ratio, 0.0, 1.0)
                
                # Convert to screen coordinates (raw)
                # Iris ratio 0.5 = center, 0.0 = left/top, 1.0 = right/bottom
                gaze_screen_x_raw = adjusted_h_ratio * MONITOR_WIDTH
                gaze_screen_y_raw = adjusted_v_ratio * MONITOR_HEIGHT
                
                # Estimate face distance for display purposes
                face_distance_cm = estimate_face_distance_cm(face_landmarks, w, h)
                
                # Apply automatic calibration (passive learning)
                gaze_screen_x, gaze_screen_y = auto_calibrator.adaptive_update(
                    gaze_screen_x_raw, gaze_screen_y_raw,
                    head_yaw, head_pitch
                )
                
                # Feed observation to calibrator during learning phase
                auto_calibrator.add_observation(
                    gaze_screen_x_raw, gaze_screen_y_raw,
                    head_yaw, head_pitch
                )
                
                # Debug output (first 5 frames to verify it's working)
                if gaze_debug_frame_count < 5:
                    print(f"[GAZE DEBUG {gaze_debug_frame_count+1}]")
                    print(f"  Iris Ratios: L_H={left_h_ratio:.3f} R_H={right_h_ratio:.3f} Avg_H={avg_h_ratio:.3f}")
                    print(f"  Iris Ratios: L_V={left_v_ratio:.3f} R_V={right_v_ratio:.3f} Avg_V={avg_v_ratio:.3f}")
                    print(f"  Head Pose: Yaw={head_yaw:.1f}° Pitch={head_pitch:.1f}° Roll={head_roll:.1f}°")
                    print(f"  Adjusted Ratios: H={adjusted_h_ratio:.3f} V={adjusted_v_ratio:.3f}")
                    print(f"  Screen Position: X={gaze_screen_x:.0f} Y={gaze_screen_y:.0f}")
                    print(f"  Distance: {face_distance_cm:.1f} cm")
                    print()
                    gaze_debug_frame_count += 1
                
                # Option 2: Smooth mouse movement to where you're looking
                pyautogui.moveTo(int(gaze_screen_x), int(gaze_screen_y), duration=0.1)
                
                # Draw gaze point on video feed (scaled to frame size)
                gaze_frame_x = int((gaze_screen_x / MONITOR_WIDTH) * w)
                gaze_frame_y = int((gaze_screen_y / MONITOR_HEIGHT) * h)
                cv2.circle(frame, (gaze_frame_x, gaze_frame_y), 8, (0, 255, 0), 2)
                cv2.circle(frame, (gaze_frame_x, gaze_frame_y), 3, (0, 255, 0), -1)
                
                # Get calibration status
                calib_status = auto_calibrator.get_status()
                
                # Create grey screen visualization showing gaze direction
                grey_viz = np.ones((400, 600, 3), dtype=np.uint8) * 128  # Grey background
                
                # Draw title
                cv2.putText(grey_viz, "Eye Gaze Direction", (180, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Calculate gaze position on visualization screen (normalized 0-1)
                gaze_norm_x = gaze_screen_x / MONITOR_WIDTH
                gaze_norm_y = gaze_screen_y / MONITOR_HEIGHT
                
                # Map to visualization window
                viz_x = int(gaze_norm_x * 600)
                viz_y = int(gaze_norm_y * 400)
                
                # Draw screen boundary
                cv2.rectangle(grey_viz, (50, 50), (550, 350), (200, 200, 200), 2)
                cv2.putText(grey_viz, "Screen Area", (250, 380),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # Draw center crosshair
                cv2.line(grey_viz, (300, 190), (300, 210), (150, 150, 150), 1)
                cv2.line(grey_viz, (290, 200), (310, 200), (150, 150, 150), 1)
                
                # Clamp gaze point to visualization bounds
                viz_x = np.clip(viz_x, 50, 550)
                viz_y = np.clip(viz_y, 50, 350)
                
                # Draw gaze point (large red circle)
                cv2.circle(grey_viz, (viz_x, viz_y), 15, (0, 0, 255), -1)
                cv2.circle(grey_viz, (viz_x, viz_y), 18, (255, 255, 255), 2)
                
                # Draw direction indicators
                direction_text = []
                if gaze_norm_x < 0.33:
                    direction_text.append("LEFT")
                elif gaze_norm_x > 0.67:
                    direction_text.append("RIGHT")
                else:
                    direction_text.append("CENTER-H")
                
                if gaze_norm_y < 0.33:
                    direction_text.append("TOP")
                elif gaze_norm_y > 0.67:
                    direction_text.append("BOTTOM")
                else:
                    direction_text.append("CENTER-V")
                
                direction_str = " + ".join(direction_text)
                cv2.putText(grey_viz, direction_str, (200, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Show coordinates
                cv2.putText(grey_viz, f"X: {gaze_screen_x:.0f} px", (20, 380),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(grey_viz, f"Y: {gaze_screen_y:.0f} px", (20, 395),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Show calibration status
                cv2.putText(grey_viz, f"Status: {calib_status}", (450, 380),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Display the grey visualization window
                cv2.imshow('Gaze Direction Visualization', grey_viz)
                
                # Display info on main video feed
                cv2.putText(frame, f"Auto Gaze: {calib_status}", (10, 220),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Gaze: ({gaze_screen_x:.0f}, {gaze_screen_y:.0f}) px", (10, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
                cv2.putText(frame, f"Distance: {face_distance_cm:.1f} cm", (10, 260),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
            # ============================================================

            left_iris_center = np.mean([[face_landmarks[idx].x * w,
                                         face_landmarks[idx].y * h]
                                        for idx in left_iris_indices], axis=0)
            right_iris_center = np.mean([[face_landmarks[idx].x * w,
                                          face_landmarks[idx].y * h]
                                         for idx in right_iris_indices], axis=0)

            cv2.circle(frame, (int(left_iris_center[0]), int(left_iris_center[1])), 5, (255, 0, 0), -1)
            cv2.circle(frame, (int(right_iris_center[0]), int(right_iris_center[1])), 5, (0, 255, 0), -1)

            if smoothed_gaze_direction != "CENTER":
                arrow_length = 50
                arrow_start = (int((left_iris_center[0] + right_iris_center[0]) / 2),
                               int((left_iris_center[1] + right_iris_center[1]) / 2))

                dx, dy = 0, 0
                if "LEFT" in smoothed_gaze_direction:
                    dx -= arrow_length
                if "RIGHT" in smoothed_gaze_direction:
                    dx += arrow_length
                if "UP" in smoothed_gaze_direction:
                    dy -= arrow_length
                if "DOWN" in smoothed_gaze_direction:
                    dy += arrow_length

                arrow_end = (arrow_start[0] + dx, arrow_start[1] + dy)
                cv2.arrowedLine(frame, arrow_start, arrow_end, (0, 255, 255), 3, tipLength=0.3)

            if smoothed_gaze_direction != last_gaze_direction_spoken:
                last_gaze_direction_spoken = smoothed_gaze_direction

            cv2.putText(frame, f"Iris Gaze: {smoothed_gaze_direction}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"L: H:{left_h_ratio:.2f} V:{left_v_ratio:.2f}", (10, 175),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(frame, f"R: H:{right_h_ratio:.2f} V:{right_v_ratio:.2f}", (10, 195),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            
            # Detect blinks for enhanced features
            blink_detected, ear = detect_blink(face_landmarks, w, h)
            
            # Detect if eyes are closed (blinking)
            is_eyes_closed = ear < blink_threshold
            
            # Detect looking up (only if enabled)
            looking_up = False
            avg_eye_ratio = 0.5
            if detect_looking_up_enabled:
                looking_up, avg_eye_ratio = detect_looking_up(face_landmarks, w, h)
                # If eyes are closed/blinking, don't report looking up
                if is_eyes_closed:
                    looking_up = False
            
            # Detect mouth open (no calibration needed!)
            mouth_open, mouth_ratio = detect_mouth_open(face_landmarks, w, h)
            
            # Print mouth state changes to command prompt with timestamps
            global last_mouth_state
            if mouth_open != last_mouth_state:
                timestamp = datetime.now().strftime("%H:%M:%S")
                if mouth_open:
                    print(f"\n[{timestamp}] >>> MOUTH OPENED <<< (Ratio: {mouth_ratio:.4f})")
                else:
                    print(f"[{timestamp}] >>> MOUTH CLOSED <<< (Ratio: {mouth_ratio:.4f})\n")
                last_mouth_state = mouth_open

            # Print looking up state changes (only if enabled and not blinking)
            if detect_looking_up_enabled:
                global last_looking_up_state
                if looking_up != last_looking_up_state and not is_eyes_closed:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    if looking_up:
                        print(f"[{timestamp}] >>> LOOKING UP <<< (Ratio: {avg_eye_ratio:.4f})")
                    else:
                        print(f"[{timestamp}] >>> LOOKING CENTER <<< (Ratio: {avg_eye_ratio:.4f})")
                    last_looking_up_state = looking_up
                elif is_eyes_closed and last_looking_up_state:
                    # Reset looking up state if eyes close
                    last_looking_up_state = False
            
            # Extract nose points for scale computation
            nose_points_3d = np.array([[face_landmarks[idx].x * w, 
                                        face_landmarks[idx].y * h,
                                        face_landmarks[idx].z * w] for idx in nose_indices])
            
            # Compute head center (still needed for cube drawing)
            head_center = np.mean(nose_points_3d, axis=0)
            
            # We still need the rotation matrix for the cube visualization
            # Note: This is a simplified "guess" for the cube only.
            # The *real* angles are from get_head_pose().
            R_final = Rscipy.from_euler('xyz', [head_pitch, head_yaw, head_roll], degrees=True).as_matrix()
            
            
            # --- START: STUDENT ADAPTIVE GESTURE LOGIC ---
            
            # 1. Get current head pose in degrees (already done by get_head_pose)
            current_yaw_deg = head_yaw
            current_pitch_deg = head_pitch
            
            # 2. Calculate relative angles based on calibration
            relative_yaw = current_yaw_deg - neutral_yaw
            relative_pitch = current_pitch_deg - neutral_pitch

            # 3. Get current time
            current_time = time.time()
            
            # 4. Get raw direction for the visual arrow
            raw_head_direction = get_head_direction(current_yaw_deg, current_pitch_deg)

            # --- State Machine ---
            if gesture_state == 'CALIBRATING':
                if current_time - calibration_start_time < CALIBRATION_DURATION:
                    # Still collecting data
                    calibration_readings_yaw.append(current_yaw_deg)
                    calibration_readings_pitch.append(current_pitch_deg)
                    last_gesture_detected = f"CALIBRATING... {CALIBRATION_DURATION - (current_time - calibration_start_time):.1f}s"
                else:
                    # Calibration finished
                    if calibration_readings_yaw: # Check if we got any readings
                        neutral_yaw = np.mean(calibration_readings_yaw)
                        neutral_pitch = np.mean(calibration_readings_pitch)
                        print("\n" + "="*60)
                        print(">>> HEAD GESTURE CALIBRATION COMPLETE")
                        print(f">>> Neutral Yaw: {neutral_yaw:.2f}°, Neutral Pitch: {neutral_pitch:.2f}°")
                        print("="*60 + "\n")
                        last_gesture_detected = "Calibration Complete!"
                        gesture_timeout_counter = 0  # Reset timeout counter after successful calibration
                    else:
                        print("\n>>> HEAD GESTURE CALIBRATION FAILED (No face detected)\n")
                        last_gesture_detected = "Calibration Failed!"
                    gesture_state = 'COOLDOWN' # Go to cooldown to allow message to be seen
                    last_gesture_time = current_time
                    gesture_persistence_counter = 0
            
            elif gesture_state == 'NEUTRAL':
                # Check for moving AWAY from neutral
                current_direction = None
                if relative_yaw > GESTURE_THRESHOLD_YAW:
                    current_direction = "RIGHT"
                elif relative_yaw < -GESTURE_THRESHOLD_YAW:
                    current_direction = "LEFT"
                elif relative_pitch > GESTURE_THRESHOLD_PITCH:
                    current_direction = "UP"
                elif relative_pitch < -GESTURE_THRESHOLD_PITCH:
                    current_direction = "DOWN"
                
                # --- NEW: Persistence Check (The "Easy Fix") ---
                if current_direction:
                    gesture_persistence_counter += 1
                    if gesture_persistence_counter >= GESTURE_PERSISTENCE_FRAMES:
                        # OK, it's a real gesture, not a tick
                        gesture_state = 'AWAY'
                        gesture_start_time = current_time
                        gesture_direction = current_direction # Lock in the first direction
                        print(f">>> GESTURE STARTED: {gesture_direction}")
                        last_gesture_detected = f"Moving {gesture_direction}..."
                        gesture_persistence_counter = 0
                else:
                    # Not away, reset the tick counter
                    gesture_persistence_counter = 0
                # --- End Persistence Check ---

            elif gesture_state == 'AWAY':
                # Check for returning to NEUTRAL
                is_neutral_now = (abs(relative_yaw) < NEUTRAL_THRESHOLD_YAW and
                                  abs(relative_pitch) < NEUTRAL_THRESHOLD_PITCH)
                
                if is_neutral_now:
                    # --- GESTURE COMPLETED! ---
                    time_elapsed = current_time - gesture_start_time
                    
                    if time_elapsed < QUICK_GESTURE_TIME:
                        gesture_type = "QUICK"
                    else:
                        gesture_type = "SLOW"
                    
                    # This is the final detected event!
                    last_gesture_detected = f"{gesture_type} {gesture_direction}"
                    print(f"\n>>> GESTURE COMPLETED: {gesture_type} {gesture_direction} (Duration: {time_elapsed:.2f}s)\n")
                    
                    # Reset timeout counter on successful gesture
                    gesture_timeout_counter = 0
                    
                    # Enter cooldown
                    gesture_state = 'COOLDOWN'
                    last_gesture_time = current_time
                    gesture_direction = None
                
                # --- NEW: Timeout Logic ---
                elif (current_time - gesture_start_time) > GESTURE_TIMEOUT:
                    # --- GESTURE FAILED (Timeout) ---
                    gesture_timeout_counter += 1
                    print(f"\n>>> GESTURE TIMEOUT: {gesture_direction} (held too long: {GESTURE_TIMEOUT}s)")
                    print(f">>> Timeout count: {gesture_timeout_counter}/{GESTURE_TIMEOUT_THRESHOLD}\n")
                    last_gesture_detected = "Gesture Failed (Timeout)"
                    
                    # Check if we need auto-recalibration
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
                        gesture_timeout_counter = 0  # Reset counter
                    else:
                        # Enter cooldown to reset the system
                        gesture_state = 'COOLDOWN'
                        last_gesture_time = current_time
                        gesture_direction = None

            elif gesture_state == 'COOLDOWN':
                # Wait for cooldown timer to finish
                if current_time - last_gesture_time > GESTURE_COOLDOWN:
                    gesture_state = 'NEUTRAL'
                    if "Calibration" not in last_gesture_detected:
                        last_gesture_detected = "" # Clear message

            # --- END: STUDENT ADAPTIVE GESTURE LOGIC ---
            
            # Draw 3D cube and axes on face to show head orientation
            if show_cube:
                # We need a proper R_final for drawing, which get_head_pose doesn't give
                # We will approximate it for the visual only
                try:
                    # This is a bit of a hack to get a matrix for drawing
                    # The angles are correct, this just makes the cube look right
                    R_yaw = _rot_y(math.radians(-head_yaw))
                    R_pitch = _rot_x(math.radians(head_pitch))
                    R_roll = _rot_y(math.radians(head_roll)) # approx
                    R_final_drawing = R_yaw @ R_pitch @ R_roll
                    
                    draw_wireframe_cube(frame, head_center, R_final_drawing, size=60)
                    draw_coordinate_axes(frame, head_center, R_final_drawing, length=80)
                except Exception as e:
                    # This might fail if R_final isn't 3x3
                    pass
            
            # Get iris landmarks
            iris_3d_left = np.mean([[face_landmarks[idx].x * w,
                                     face_landmarks[idx].y * h,
                                     face_landmarks[idx].z * w] for idx in left_iris_indices], axis=0)
            iris_3d_right = np.mean([[face_landmarks[idx].x * w,
                                      face_landmarks[idx].y * h,
                                      face_landmarks[idx].z * w] for idx in right_iris_indices], axis=0)
            
            # Draw face mesh
            for lm in face_landmarks:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)
            
            # Calculate gaze if calibrated
            if left_sphere_locked and right_sphere_locked:
                # Gaze calculation needs a solid R_final
                # Let's use the one from get_head_pose
                R_final_gaze = Rscipy.from_euler('xyz', [head_pitch, -head_yaw, head_roll], degrees=True).as_matrix()
                
                # Compute scale-aware sphere positions
                current_nose_scale = compute_scale(nose_points_3d)
                scale_ratio_l = current_nose_scale / left_calibration_nose_scale if left_calibration_nose_scale else 1.0
                scale_ratio_r = current_nose_scale / right_calibration_nose_scale if right_calibration_nose_scale else 1.0
                
                sphere_world_l = head_center + R_final_gaze @ (left_sphere_local_offset * scale_ratio_l)
                sphere_world_r = head_center + R_final_gaze @ (right_sphere_local_offset * scale_ratio_r)
                
                # Calculate gaze directions
                left_gaze_dir = iris_3d_left - sphere_world_l
                left_gaze_dir /= np.linalg.norm(left_gaze_dir) + 1e-9
                right_gaze_dir = iris_3d_right - sphere_world_r
                right_gaze_dir /= np.linalg.norm(right_gaze_dir) + 1e-9
                
                # Combined gaze direction
                combined_direction = (left_gaze_dir + right_gaze_dir) / 2
                combined_direction /= np.linalg.norm(combined_direction) + 1e-9
                
                # Add to smoothing buffer
                combined_gaze_directions.append(combined_direction)
                
                # Average for smoothing
                if len(combined_gaze_directions) > 0:
                    avg_combined_direction = np.mean(combined_gaze_directions, axis=0)
                    avg_combined_direction /= np.linalg.norm(avg_combined_direction) + 1e-9
                    
                    # Convert to screen coordinates and get yaw/pitch
                    screen_x, screen_y, yaw, pitch = convert_gaze_to_screen_coordinates(
                        avg_combined_direction, calibration_offset_yaw, calibration_offset_pitch
                    )
                    
                    # Draw gaze point on frame
                    gaze_x = int((screen_x / MONITOR_WIDTH) * w)
                    gaze_y = int((screen_y / MONITOR_HEIGHT) * h)
                    cv2.circle(frame, (gaze_x, gaze_y), 10, (0, 0, 255), 2)
                    cv2.line(frame, (int(iris_3d_left[0]), int(iris_3d_left[1])), 
                             (gaze_x, gaze_y), (255, 0, 0), 1)
                    cv2.line(frame, (int(iris_3d_right[0]), int(iris_3d_right[1])), 
                             (gaze_x, gaze_y), (0, 255, 0), 1)
                    
                    # Update mouse if enabled
                    if mouse_control_enabled:
                        with mouse_lock:
                            mouse_target[0] = screen_x
                            mouse_target[1] = screen_y
                        pyautogui.moveTo(screen_x, screen_y)
                    
                    # Send to OptiKey if enabled
                    # TODO: Implement send_to_optikey function if OptiKey integration needed
                    # if optikey_enabled:
                    #     send_to_optikey(screen_x, screen_y)
                    
                    # Write screen position
                    write_screen_position(screen_x, screen_y)
            
            gaze_overlay_direction = smoothed_gaze_direction
            
            # Draw metrics with gaze and head direction
            draw_eye_metrics(frame, ear, gaze_overlay_direction, raw_head_direction, 
                             head_yaw, head_pitch, head_roll, 
                             last_gesture_detected, gesture_state,
                             looking_up, avg_eye_ratio,
                             mouth_open, mouth_ratio)
            
            # Update orbit controls
            update_orbit_from_keys()
        else:
            # No face detected
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            # Pass new gesture info
            draw_eye_metrics(frame, 0, gaze_overlay_direction, None, 
                             0, 0, 0, 
                             "No face detected", "NEUTRAL",
                             False, 0.5,
                             False, 0.0)
        
        # Apply high contrast if enabled
        if high_contrast_mode:
            frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=20)
        
        cv2.imshow("Enhanced Eye Tracking", frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('c') and not (left_sphere_locked and right_sphere_locked):
            # Calibrate eye spheres
            if results.multi_face_landmarks:
                R_final_calib = Rscipy.from_euler('xyz', [head_pitch, -head_yaw, head_roll], degrees=True).as_matrix()
                current_nose_scale = compute_scale(nose_points_3d)
                base_radius = 30  # Base sphere radius
                
                # Lock LEFT eye
                left_sphere_local_offset = R_final_calib.T @ (iris_3d_left - head_center)
                camera_dir_world = np.array([0, 0, 1])
                camera_dir_local = R_final_calib.T @ camera_dir_world
                left_sphere_local_offset += base_radius * camera_dir_local
                left_calibration_nose_scale = current_nose_scale
                left_sphere_locked = True
                
                # Lock RIGHT eye
                right_sphere_local_offset = R_final_calib.T @ (iris_3d_right - head_center)
                right_sphere_local_offset += base_radius * camera_dir_local
                right_calibration_nose_scale = current_nose_scale
                right_sphere_locked = True
                
                # Create 3D monitor plane
                sphere_world_l_calib = head_center + R_final_calib @ left_sphere_local_offset
                sphere_world_r_calib = head_center + R_final_calib @ right_sphere_local_offset
                
                left_dir = iris_3d_left - sphere_world_l_calib
                right_dir = iris_3d_right - sphere_world_r_calib
                
                if np.linalg.norm(left_dir) > 1e-9: left_dir /= np.linalg.norm(left_dir)
                if np.linalg.norm(right_dir) > 1e-9: right_dir /= np.linalg.norm(right_dir)
                
                forward_hint = (left_dir + right_dir) * 0.5
                if np.linalg.norm(forward_hint) > 1e-9:
                    forward_hint /= np.linalg.norm(forward_hint)
                else:
                    forward_hint = None
                
                gaze_origin = (sphere_world_l_calib + sphere_world_r_calib) / 2
                gaze_dir = forward_hint
                
                monitor_corners, monitor_center_w, monitor_normal_w, units_per_cm = create_monitor_plane(
                    head_center, R_final_calib, face_landmarks, w, h,
                    forward_hint=forward_hint,
                    gaze_origin=gaze_origin,
                    gaze_dir=gaze_dir
                )
                
                debug_world_frozen = True
                orbit_pivot_frozen = monitor_center_w.copy()
                
                print("[Calibration Complete] Both eye spheres locked")

        elif key == ord('g'):
            if results.multi_face_landmarks:
                # Use ROI-based method if enabled, otherwise fallback
                if use_roi_processing:
                    left_eye_roi = extract_eye_roi_info(
                        face_landmarks, LEFT_EYE_LANDMARKS_71, w, h, padding_ratio=0.2
                    )
                    right_eye_roi = extract_eye_roi_info(
                        face_landmarks, RIGHT_EYE_LANDMARKS_71, w, h, padding_ratio=0.2
                    )
                    
                    if left_eye_roi and right_eye_roi:
                        left_h, left_v = get_iris_position_ratio_roi(
                            face_landmarks, left_iris_indices, left_eye_roi, w, h
                        )
                        right_h, right_v = get_iris_position_ratio_roi(
                            face_landmarks, right_iris_indices, right_eye_roi, w, h
                        )
                    else:
                        # Fallback
                        left_h, left_v = get_iris_position_ratio(
                            face_landmarks, LEFT_EYE_CORNERS, LEFT_EYE_LIDS,
                            left_iris_indices, w, h,
                            eye_contour_indices=LEFT_EYE_LANDMARKS_71
                        )
                        right_h, right_v = get_iris_position_ratio(
                            face_landmarks, RIGHT_EYE_CORNERS, RIGHT_EYE_LIDS,
                            right_iris_indices, w, h,
                            eye_contour_indices=RIGHT_EYE_LANDMARKS_71
                        )
                else:
                    # Use improved 71-landmark method for calibration
                    left_h, left_v = get_iris_position_ratio(
                        face_landmarks, LEFT_EYE_CORNERS, LEFT_EYE_LIDS,
                        left_iris_indices, w, h,
                        eye_contour_indices=LEFT_EYE_LANDMARKS_71
                    )
                    right_h, right_v = get_iris_position_ratio(
                        face_landmarks, RIGHT_EYE_CORNERS, RIGHT_EYE_LIDS,
                        right_iris_indices, w, h,
                        eye_contour_indices=RIGHT_EYE_LANDMARKS_71
                    )

                iris_calibration_h_offset = 0.5 - ((left_h + right_h) / 2.0)
                iris_calibration_v_offset = 0.5 - ((left_v + right_v) / 2.0)

                print(f"[Iris Calibration] H offset: {iris_calibration_h_offset:.3f}, V offset: {iris_calibration_v_offset:.3f}")
                if use_roi_processing:
                    print("[Using ROI-based processing]")
        
        # === NEW: 'k' key for Head Gesture Calibration ===
        elif key == ord('k'):
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
                gesture_timeout_counter = 0  # Reset timeout counter when manually calibrating
            else:
                print(">>> Calibration already in progress...")
        # =================================================
            
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
        elif key == ord('u'):
            detect_looking_up_enabled = not detect_looking_up_enabled
            print(f"Looking up detection: {'ON' if detect_looking_up_enabled else 'OFF'}")
        elif key == ord('t'):
            use_roi_processing = not use_roi_processing
            if use_roi_processing:
                left_eye_ratio_buffer = deque(maxlen=5)
                right_eye_ratio_buffer = deque(maxlen=5)
            print(f"ROI-based eye processing: {'ON' if use_roi_processing else 'OFF'}")
        elif key == ord('y'):
            use_hybrid_processing = not use_hybrid_processing
            if use_hybrid_processing:
                # Initialize Kalman filters
                if KALMAN_AVAILABLE:
                    from filterpy.kalman import KalmanFilter
                    left_eye_kalman = KalmanFilter(dim_x=2, dim_z=2)
                    left_eye_kalman.x = np.array([0.5, 0.5])
                    left_eye_kalman.F = np.eye(2)
                    left_eye_kalman.H = np.eye(2)
                    left_eye_kalman.P *= 0.1
                    left_eye_kalman.R *= 0.1
                    left_eye_kalman.Q *= 0.01
                    
                    right_eye_kalman = KalmanFilter(dim_x=2, dim_z=2)
                    right_eye_kalman.x = np.array([0.5, 0.5])
                    right_eye_kalman.F = np.eye(2)
                    right_eye_kalman.H = np.eye(2)
                    right_eye_kalman.P *= 0.1
                    right_eye_kalman.R *= 0.1
                    right_eye_kalman.Q *= 0.01
                    print("[Hybrid Mode] Using advanced Kalman filter")
                else:
                    left_eye_kalman = SimpleKalmanFilter()
                    right_eye_kalman = SimpleKalmanFilter()
                    print("[Hybrid Mode] Using simple Kalman filter (install filterpy for advanced)")
                # Ensure ROI processing is enabled for hybrid
                if not use_roi_processing:
                    use_roi_processing = True
                    left_eye_ratio_buffer = deque(maxlen=5)
                    right_eye_ratio_buffer = deque(maxlen=5)
                    print("[Hybrid Mode] Enabled ROI processing (required)")
            else:
                left_eye_kalman = None
                right_eye_kalman = None
            print(f"Hybrid processing (MediaPipe + OpenCV + Kalman): {'ON' if use_hybrid_processing else 'OFF'}")
        elif key == ord('a'):
            use_auto_gaze = not use_auto_gaze
            if use_auto_gaze:
                # Reinitialize calibrator when toggling on
                auto_calibrator = AutoGazeCalibrator(
                    adaptation_rate=0.02,
                    initial_observation_time=10.0
                )
                print(f"AUTO GAZE tracking: ON (No calibration needed!)")
                print(f"  - System will learn passively for {auto_calibrator.initial_observation_time}s")
                print(f"  - Just look naturally at the screen")
            else:
                print(f"AUTO GAZE tracking: OFF")
        elif key == ord('x'):
            # Place marker (simplified for now)
            if left_sphere_locked and right_sphere_locked:
                print("[Marker] Placed at current gaze position")
        
        # Function key handling
        if keyboard.is_pressed('f7'):
            mouse_control_enabled = not mouse_control_enabled
            print(f"Mouse control: {'ON' if mouse_control_enabled else 'OFF'}")
            time.sleep(0.3)
        elif keyboard.is_pressed('f8'):
            optikey_enabled = not optikey_enabled
            print(f"OptiKey control: {'ON' if optikey_enabled else 'OFF'}")
            time.sleep(0.3)
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\nEye tracker closed")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()
