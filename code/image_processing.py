import cv2
import struct
import socket
import numpy as np
import mediapipe as mp 
from mediapipe.tasks import python  
from mediapipe.tasks.python import vision
from mediapipe import solutions     #if error occurs you need to install the mediapipe 0.10.9 version
from mediapipe.framework.formats import landmark_pb2
import math
import time
import requests
import threading
import math

import os

# Get the absolute path and normalize it
BASE_DIR = r"D:\Facultate\VedemCeIese\pi-p-proiect-meowtrix\Models"
HAND_MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")
POSE_MODEL_PATH = os.path.join(BASE_DIR, "pose_landmarker_full.task")

print(f"paths: {HAND_MODEL_PATH}\n {POSE_MODEL_PATH}")

# Double check if the files actually exist to avoid later errors
if not os.path.exists(HAND_MODEL_PATH):
    print(f"ERROR: Hand model not found at {HAND_MODEL_PATH}")

# HAND_MODEL_PATH = "D:\Facultate\VedemCeIese\pi-p-proiect-meowtrix\Models\hand_landmarker.task"
# POSE_MODEL_PATH = "D:\Facultate\VedemCeIese\pi-p-proiect-meowtrix\Models\pose_landmarker_full.task"

HOST = "0.0.0.0"
PORT = 5001

NAO_BRIDGE_URL = "http://127.0.0.1:5050/movement"

COMMAND_INTERVAL = 0.7  # 10 comenzi pe secundă max
LAST_COMMAND_TIME = 0

# Inițializare MediaPipe Pose (SIMPLU)
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

    
def right_hand_joints(detection_result):
    NAO_LIMITS = {
    "LShoulderPitch": [-2.0857, 2.0857],
    "LShoulderRoll":  [-0.3142, 1.3265],
    "LElbowYaw":      [-2.0857, 2.0857],
    "LElbowRoll":     [-1.5446, -0.0349]
}

    pose_landmarks_list = detection_result.pose_world_landmarks[0]

    # mediapipe pose_landmarks
    Rshoulder = pose_landmarks_list[12]
    Lshoulder = pose_landmarks_list[11] 
    Relbow    = pose_landmarks_list[14] 
    Rhip      = pose_landmarks_list[24]  

    # Calcul puncte
    Rshoulder_Point = np.array([Rshoulder.x, Rshoulder.y, Rshoulder.z])
    Lshoulder_Point = np.array([Lshoulder.x, Lshoulder.y, Lshoulder.z])
    Relbow_Point    = np.array([Relbow.x, Relbow.y, Relbow.z])
    Rhip_Point      = np.array([Rhip.x, Rhip.y, Rhip.z])

    # Calcul X, Y, Z - referinta
    X = Lshoulder_Point - Rshoulder_Point
    X /= np.linalg.norm(X)

    Y = Rhip_Point - Rshoulder_Point
    Y /= np.linalg.norm(Y)

    Z = np.cross(Y, X)
    Z /= np.linalg.norm(Z)

    # Vector brat
    Varm = Relbow_Point - Rshoulder_Point
    Varm /= np.linalg.norm(Varm)

    L_Shoulder_Pitch = np.atan2(np.dot(Varm, Y), np.dot(Varm, Z))

    Varm_in_plane = Varm - np.dot(Varm, X) * X 
    Varm_in_plane /= np.linalg.norm(Varm_in_plane)

    L_Shoulder_Roll = -np.atan2(np.dot(Varm, X), np.sqrt(np.dot(Varm, Y)**2 + np.dot(Varm, Z)**2))

    #Clamping
    L_Shoulder_Pitch = np.clip(L_Shoulder_Pitch, NAO_LIMITS["LShoulderPitch"][0], NAO_LIMITS["LShoulderPitch"][1])
    L_Shoulder_Roll = np.clip(L_Shoulder_Roll, NAO_LIMITS["LShoulderRoll"][0], NAO_LIMITS["LShoulderRoll"][1])

    return L_Shoulder_Pitch, L_Shoulder_Roll  



def send_arm_angles(joints):
    """Trimite unghiurile către serverul Flask în thread separat."""
    def _send():
        try:
            pitch, roll = joints
            payload = {
                "type": "joints",
                "joints": {
                    "LShoulderPitch": pitch,
                    "LShoulderRoll": roll
                    }

                }
            requests.post(NAO_BRIDGE_URL, json=payload, timeout=1.00)
            print(f"SENT: {payload}")
        except Exception as e:
            print(f"EXCEPTIE: {e}")
    threading.Thread(target=_send, daemon=True).start()

def draw_pose_landmarks_on_image(rgb_image, detection_result):
    """Desenează landmark-urile pose pe imagine."""
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) 
            for landmark in pose_landmarks
        ])

        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
            
    return annotated_image

def draw_hand_landmarks_on_image(rgb_image, detection_result):
    """Desenează landmark-urile mâinii pe imagine."""
    MARGIN = 10
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    HANDEDNESS_TEXT_COLOR = (88, 205, 54)

    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) 
            for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())

    return annotated_image

def camera_capture(camera_input):
    """Funcția principală pentru captură video și mirror game."""
    global LAST_COMMAND_TIME
    
    conn = None
    srv = None
    pose_landmarker = None
    hand_landmarker = None
    camera = None

    if camera_input not in ["NAO", "laptop"]:
        raise Exception("Wrong camera_input. Use 'NAO' or 'laptop'")

    if camera_input == "NAO":
        srv = setup_server(HOST, PORT)
        print("Waiting for NAO to connect...")
        conn, addr = srv.accept()
        print(f"Connected by {addr}")
    
    try:

        # Load the binary data from the files (because mediapipe wouldnt fucking find the path correctly for some reason)
        with open(HAND_MODEL_PATH, 'rb') as f:
            hand_model_data = f.read()
        with open(POSE_MODEL_PATH, 'rb') as f:
            pose_model_data = f.read()


        if camera_input == "laptop":
            camera = cv2.VideoCapture(0)

        # Inițializări MediaPipe (versiunea veche cu tasks)
        PoseBaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        PoseVisionRunningMode = mp.tasks.vision.RunningMode

        HandBaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        HandVisionRunningMode = mp.tasks.vision.RunningMode

        hand_options = HandLandmarkerOptions(
            base_options=HandBaseOptions(model_asset_buffer=hand_model_data),
            running_mode=HandVisionRunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
      
        pose_options = PoseLandmarkerOptions(
            base_options=PoseBaseOptions(model_asset_buffer=pose_model_data),
            running_mode=PoseVisionRunningMode.IMAGE,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=True
        )
      
        hand_landmarker = HandLandmarker.create_from_options(hand_options)
        pose_landmarker = PoseLandmarker.create_from_options(pose_options)

        print("\n=== NAO MIRROR GAME ===")
        print("Mișcă brațul drept - NAO te urmează!")
        print("Apasă 'q' pentru ieșire\n")

        while True:
            # Primește frame
            if camera_input == "NAO":
                frame = receive_frame(conn)
                if frame is None:
                    print("Client disconnected.")
                    break
            elif camera_input == "laptop":
                ret, frame = camera.read()
                if not ret:
                    break

            # Procesare MediaPipe
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

            # Detecție
            pose_landmarker_result = pose_landmarker.detect(mp_image)
            hand_landmarker_result = hand_landmarker.detect(mp_image)

            # Vizualizare
            annotated_image_rgb = draw_pose_landmarks_on_image(image_rgb, pose_landmarker_result)
            annotated_image_rgb = draw_hand_landmarks_on_image(annotated_image_rgb, hand_landmarker_result)
            annotated_image_bgr = cv2.cvtColor(annotated_image_rgb, cv2.COLOR_RGB2BGR)

            # === MIRROR GAME LOGIC ===
            if pose_landmarker_result.pose_landmarks:
                try:
                    # Calculează joints in radiani
                    joints = right_hand_joints(pose_landmarker_result)
                    
                    # Afișează pe ecran
                    # y = 30
                    # for nume, val in joints.items():
                    #     text = f"{nume}: {math.degrees(val):.1f}°"
                    #     cv2.putText(annotated_image_bgr, text, (10, y),
                    #                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    #     y += 25
                    
                    # Trimite la NAO (cu rate limiting)
                    timp_acum = time.time()
                    if timp_acum - LAST_COMMAND_TIME > COMMAND_INTERVAL:
                        send_arm_angles(joints)
                        LAST_COMMAND_TIME = timp_acum
                        
                except Exception as e:
                    print(f"Eroare calcul unghiuri: {e}\n\n")

            # Afișare
            cv2.imshow("NAO Mirror Game", annotated_image_bgr)

            # Ieșire cu 'q'
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break 

    except Exception as e:
        print(f"Eroare: {e}")
    finally:
        if conn:
            conn.close()
        if srv:
            srv.close()
        if pose_landmarker:
            pose_landmarker.close()
        if hand_landmarker:
            hand_landmarker.close()
        if camera:
            camera.release()
        cv2.destroyAllWindows()

def setup_server(host, port):
    """Creează server socket pentru primire frame-uri de la NAO."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    print(f"Server listening on {host}:{port}")
    return srv

def recv_all(sock, size):
    """Helper pentru primirea completă a datelor."""
    data = b""
    while len(data) < size:
        packet = sock.recv(size - len(data))
        if not packet:
            return None
        data += packet
    return data

def receive_frame(conn):
    """Primește frame de la NAO via socket."""
    header = recv_all(conn, 4)
    if not header:
        return None
    length = struct.unpack("!I", header)[0]
    
    payload = recv_all(conn, length)
    if not payload:
        return None

    frame = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    return frame

def main():
    # Schimbă între "laptop" și "NAO"
    camera_capture("NAO")

if __name__ == '__main__':
    main()
