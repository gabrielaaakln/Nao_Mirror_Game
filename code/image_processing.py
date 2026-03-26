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
    min_detection_confidence=1.0,
    min_tracking_confidence=1.0
)

def right_hand_joints(detection_result, right_hand_landmarks):
    NAO_LIMITS = {
        "LShoulderPitch": [-2.0857, 2.0857],
        "LShoulderRoll":  [-0.3142, 1.3265],
        "LElbowYaw":      [-2.0857, 2.0857],
        "LElbowRoll":     [-1.5446, -0.0349],
        "LWristYaw":      [-1.8238, 1.8238],
        "LHand":          [0.0, 1.0]
    }

    pose_landmarks_list = detection_result.pose_world_landmarks[0]

    # Extragem landmark-urile pentru brațul DREPT al omului
    Rshoulder_Point = np.array([pose_landmarks_list[12].x, pose_landmarks_list[12].y, pose_landmarks_list[12].z])
    Lshoulder_Point = np.array([pose_landmarks_list[11].x, pose_landmarks_list[11].y, pose_landmarks_list[11].z])
    Relbow_Point    = np.array([pose_landmarks_list[14].x, pose_landmarks_list[14].y, pose_landmarks_list[14].z])
    RWrist_Point    = np.array([pose_landmarks_list[16].x, pose_landmarks_list[16].y, pose_landmarks_list[16].z])
    Rhip_Point      = np.array([pose_landmarks_list[24].x, pose_landmarks_list[24].y, pose_landmarks_list[24].z])

    # Sistem de referință trunchi (Om)
    X = Lshoulder_Point - Rshoulder_Point # Spre stânga omului
    X /= np.linalg.norm(X)

    Y = Rhip_Point - Rshoulder_Point      # În jos
    Y /= np.linalg.norm(Y)

    Z = np.cross(Y, X)                    # Spre cameră / În față
    Z /= np.linalg.norm(Z)

    # Vectori braț și antebraț (Om)
    Varm = Relbow_Point - Rshoulder_Point
    Varm /= np.linalg.norm(Varm)

    Vforearm = RWrist_Point - Relbow_Point
    Vforearm /= np.linalg.norm(Vforearm)

    # 1. Calculăm Shoulder Pitch și Roll (Logica ta funcționează perfect pentru mirroring)
    L_Shoulder_Pitch = np.arctan2(np.dot(Varm, Y), np.dot(Varm, Z))
    L_Shoulder_Roll  = -np.arctan2(np.dot(Varm, X), np.sqrt(np.dot(Varm, Y)**2 + np.dot(Varm, Z)**2))

    # IMPORTANT: Facem clamp la umăr ÎNAINTE de a calcula cotul. 
    # Dacă NAO nu poate atinge unghiul tău de umăr, referința pentru cot trebuie calculată
    # de la poziția fizică în care se află brațul robotului, nu al tău.
    L_Shoulder_Pitch = np.clip(L_Shoulder_Pitch, *NAO_LIMITS["LShoulderPitch"])
    L_Shoulder_Roll  = np.clip(L_Shoulder_Roll,  *NAO_LIMITS["LShoulderRoll"])

    # 2. Elbow Roll
    L_Elbow_Roll = -np.arccos(np.clip(np.dot(Varm, Vforearm), -1.0, 1.0))

    # 3. Elbow Yaw - Mapare pe cinematica robotului NAO
    # Mapăm direcțiile brațului omului direct pe sistemul de coordonate stâng de la NAO:
    # X_nao (Față) = Z_om, Y_nao (Stânga) = -X_om, Z_nao (Sus) = -Y_om
    Varm_nao = np.array([np.dot(Varm, Z), -np.dot(Varm, X), -np.dot(Varm, Y)])
    Vforearm_nao = np.array([np.dot(Vforearm, Z), -np.dot(Vforearm, X), -np.dot(Vforearm, Y)])

    # Normala planului format de braț și antebraț
    n_arm = np.cross(Varm_nao, Vforearm_nao)
    norm_n = np.linalg.norm(n_arm)

    if norm_n > 1e-4: # Brațul nu este complet întins (gimbal lock natural)
        n_arm /= norm_n

        # Reconstruim orientarea umărului robotului folosind unghiurile calculate
        cp, sp = np.cos(L_Shoulder_Pitch), np.sin(L_Shoulder_Pitch)
        cr, sr = np.cos(L_Shoulder_Roll),  np.sin(L_Shoulder_Roll)

        # Deduse din înmulțirea matricilor RotY(Pitch) * RotZ(Roll)
        # Acestea sunt axele locale Y și Z ale brațului superior DUPĂ rotirea din umăr
        Y_local = np.array([-cp*sr, cr, sp*sr]) 
        Z_local = np.array([sp, 0, cp])         

        # Calculăm Yaw proiectând normala brațului pe axele locale
        L_Elbow_Yaw = np.arctan2(np.dot(n_arm, Y_local), -np.dot(n_arm, Z_local))
    else:
        # Dacă brațul e perfect întins, Yaw-ul este incalculabil geometric. Păstrăm 0.
        L_Elbow_Yaw = 0.0

    # 3.5. WRIST YAW - Referință stabilă bazată pe trunchi

    # landmark 5 = INDEX_MCP, landmark 17 = PINKY_MCP (baza degetelor, foarte stabile)
    index_mcp = np.array([right_hand_landmarks[5].x, right_hand_landmarks[5].y])
    pinky_mcp = np.array([right_hand_landmarks[17].x, right_hand_landmarks[17].y])

    dx = index_mcp[0] - pinky_mcp[0]
    dy = index_mcp[1] - pinky_mcp[1]

    raw_wrist_angle = np.arctan2(dy, dx)
    L_Wrist_Yaw = raw_wrist_angle + np.pi/2

    # HAND OPEN/CLOSE
    FINGERTIPS = [8, 12, 16, 20]
    FINGER_MCP  = [5,  9, 13, 17]
    wrist = np.array([right_hand_landmarks[0].x,
                      right_hand_landmarks[0].y,
                      right_hand_landmarks[0].z])
    total_score = 0.0
    for tip_idx, mcp_idx in zip(FINGERTIPS, FINGER_MCP):
        tip = np.array([right_hand_landmarks[tip_idx].x,
                        right_hand_landmarks[tip_idx].y,
                        right_hand_landmarks[tip_idx].z])
        mcp = np.array([right_hand_landmarks[mcp_idx].x,
                        right_hand_landmarks[mcp_idx].y,
                        right_hand_landmarks[mcp_idx].z])
        dist_tip = np.linalg.norm(tip - wrist)
        dist_mcp = np.linalg.norm(mcp - wrist)
        total_score += dist_tip / (dist_mcp + 1e-6)

    avg_score = total_score / len(FINGERTIPS)
    L_Hand = float(np.clip((avg_score - 1.0) / 1.5, 0.0, 1.0))

    # Limităm unghiurile
    L_Elbow_Roll     = np.clip(L_Elbow_Roll,     *NAO_LIMITS["LElbowRoll"])
    L_Elbow_Yaw      = np.clip(L_Elbow_Yaw,      *NAO_LIMITS["LElbowYaw"])
    L_Wrist_Yaw      = np.clip(L_Wrist_Yaw,      *NAO_LIMITS["LWristYaw"])

    return L_Shoulder_Pitch, L_Shoulder_Roll, L_Elbow_Roll, L_Elbow_Yaw, L_Wrist_Yaw, L_Hand


def send_arm_angles(joints):
    """Trimite unghiurile către serverul Flask în thread separat."""
    def _send():
        try:
            shoulder_pitch, shoulder_roll, elbow_roll, elbow_yaw, wrist_yaw, hand = joints
            payload = {
                "type": "joints",
                "joints": {
                    "LShoulderPitch": shoulder_pitch,
                    "LShoulderRoll": shoulder_roll,
                    "LElbowRoll": elbow_roll,
                    "LElbowYaw": elbow_yaw,
                    "LWristYaw": wrist_yaw,
                    "LHand": hand
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
            # === MIRROR GAME LOGIC ===
            if pose_landmarker_result.pose_landmarks:
                try:
                    # Găsește mâna dreaptă din hand results
                    right_hand_landmarks = None
                    for i, handedness in enumerate(hand_landmarker_result.handedness):
                        if handedness[0].category_name == "Right":
                            right_hand_landmarks = hand_landmarker_result.hand_landmarks[i]
                            break

                    # Calculează joints doar dacă avem și mâna detectată
                    if right_hand_landmarks is not None:
                        joints = right_hand_joints(pose_landmarker_result, right_hand_landmarks)
            
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
    camera_capture("laptop")

if __name__ == '__main__':
    main()
