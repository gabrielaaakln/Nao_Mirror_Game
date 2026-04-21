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
import json
import os
import jointsCalculator
import json, statistics

# Get the absolute path and normalize it
# BASE_DIR = r"D:\Facultate\VedemCeIese\pi-p-proiect-meowtrix\Models"
# HAND_MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")
# POSE_MODEL_PATH = os.path.join(BASE_DIR, "pose_landmarker_full.task")

HAND_MODEL_PATH = '/Users/ziza/University/an3/Sem1/PI-P_utils/nao-docker-bridge/hand_landmarker.task'
POSE_MODEL_PATH = '/Users/ziza/University/an3/Sem1/PI-P_utils/nao-docker-bridge/pose_landmarker_heavy.task'

print(f"paths: {HAND_MODEL_PATH}\n {POSE_MODEL_PATH}")

# Double check if the files actually exist to avoid later errors
if not os.path.exists(HAND_MODEL_PATH):
    print(f"ERROR: Hand model not found at {HAND_MODEL_PATH}")

# HAND_MODEL_PATH = "D:\Facultate\VedemCeIese\pi-p-proiect-meowtrix\Models\hand_landmarker.task"
# POSE_MODEL_PATH = "D:\Facultate\VedemCeIese\pi-p-proiect-meowtrix\Models\pose_landmarker_full.task"

HOST = "0.0.0.0"
PORT = 5001

NAO_BRIDGE_URL = "http://127.0.0.1:5050/movement"

COMMAND_INTERVAL = 0.05  # comenzi la 20fps
LAST_COMMAND_TIME = 0

NAO_CMD_IP = "127.0.0.1" # Ensure this points to your command bridge Docker IP
NAO_CMD_PORT = 5050
udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Inițializare MediaPipe Pose (SIMPLU)
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    min_detection_confidence=1.0,
    min_tracking_confidence=1.0
)

def send_head_angles(HeadJoints):
    """Fires a UDP packet to update the head joints."""
    try:
        head_yaw, head_pitch = HeadJoints
        payload = {
            "type": "joints",
            "HeadJoints": {
                "HeadYaw": head_yaw,
                "HeadPitch": head_pitch
            }
        }
        udp_sock.sendto(json.dumps(payload).encode('utf-8'), (NAO_CMD_IP, NAO_CMD_PORT))
    except Exception as e:
        print(f"UDP Error (Head): {e}")

def send_Larm_angles(LHandjoints):
    """Fires a UDP packet to the command bridge instantly."""
    try:
        LshoulderPitch, LshoulderRoll, LelbowRoll, LelbowYaw, Lwrist_yaw, Lhand = LHandjoints
        payload = {
            "type": "joints",
            "LArmJoints": {
                "LShoulderPitch": float(LshoulderPitch),
                "LShoulderRoll": float(LshoulderRoll),
                "LElbowRoll": float(LelbowRoll),
                "LElbowYaw": float(LelbowYaw),
                "LWristYaw": float(Lwrist_yaw),
                "LHand": float(Lhand)
            }
        }
        # Fire and forget. No waiting for a response.
        udp_sock.sendto(json.dumps(payload).encode('utf-8'), (NAO_CMD_IP, NAO_CMD_PORT))
    except Exception as e:
        print(f"UDP Error: {e}")



def send_Rarm_angles(RHandjoints):
    """Fires a UDP packet to the command bridge instantly."""
    try:
        RshoulderPitch, RshoulderRoll, RelbowRoll, RelbowYaw, Rwrist_yaw, Rhand = RHandjoints

        payload = {
            "type": "joints",
            "RArmJoints": {
                "RShoulderPitch": float(RshoulderPitch),
                "RShoulderRoll": float(RshoulderRoll),
                "RElbowRoll": float(RelbowRoll),
                "RElbowYaw": float(RelbowYaw),
                "RWristYaw": float(Rwrist_yaw),
                "RHand": float(Rhand)
            }
        }
        # Fire and forget. No waiting for a response.
        udp_sock.sendto(json.dumps(payload).encode('utf-8'), (NAO_CMD_IP, NAO_CMD_PORT))
    except Exception as e:
        print(f"UDP Error: {e}")

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

    total_frames = 0
    pose_detected_frames = 0
    hand_detected_frames = 0
    inference_log = [] 
    latency_log = []   

    if camera_input not in ["NAO", "laptop"]:
        raise Exception("Wrong camera_input. Use 'NAO' or 'laptop'")

    if camera_input == "NAO":
        # We just bind the socket. No waiting for accept().
        srv = setup_server(HOST, PORT)
        print("Waiting for NAO to connect...")
        conn, addr = srv.accept()  # Pass the server socket directly to receive_frame
        # Ensure the accepted connection also disables buffering
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1) 
    
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
            frame_start = time.time()

            
            # Primește frame
            if camera_input == "NAO":
                frame = receive_frame(conn)
                if frame is None:
                    break

            elif camera_input == "laptop":
                ret, frame = camera.read()
                if not ret:
                    break

            total_frames += 1
            
            # Procesare MediaPipe
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

            # Detecție
            t0 = time.time()
            pose_landmarker_result = pose_landmarker.detect(mp_image)
            hand_landmarker_result = hand_landmarker.detect(mp_image)
            inference_ms = (time.time() - t0) * 1000
            inference_log.append(inference_ms)

            if pose_landmarker_result.pose_landmarks:
                pose_detected_frames += 1
            if hand_landmarker_result.hand_landmarks:
                hand_detected_frames += 1

            # Vizualizare
            annotated_image_rgb = draw_pose_landmarks_on_image(image_rgb, pose_landmarker_result)
            annotated_image_rgb = draw_hand_landmarks_on_image(annotated_image_rgb, hand_landmarker_result)
            annotated_image_bgr = cv2.cvtColor(annotated_image_rgb, cv2.COLOR_RGB2BGR)

            # === MIRROR GAME LOGIC ===
            if pose_landmarker_result.pose_landmarks:
                try:
                    # Găsește mâna dreaptă din hand results
                    # Reprezentăm mâinile fizice ale OMULUI
                    human_right_hand_landmarks = None
                    human_left_hand_landmarks = None
                    
                    # Reprezentăm comenzile finale pentru NAO
                    nao_left_arm_joints = None
                    nao_right_arm_joints = None

                    # Găsește mâna dreaptă din hand results
                    for i, handedness in enumerate(hand_landmarker_result.handedness):
                        label = handedness[0].category_name
                        landmarks = hand_landmarker_result.hand_landmarks[i]
                        
                        # Assuming your camera mirror logic is correct:
                        if label == "Right": 
                            human_right_hand_landmarks = landmarks
                            break
                        elif label == "Left":   
                            human_left_hand_landmarks = landmarks
                            break

                    # MIRRORING LOGIC:
                    # Omul mișcă mâna DREAPTĂ -> Calculează Pose Dreapta -> Mută brațul STÂNG al lui NAO
                    nao_left_arm_joints = jointsCalculator.right_hand_joints(pose_landmarker_result, human_right_hand_landmarks)
                    if nao_left_arm_joints is not None:
                        jointsCalculator.log_stability("L", nao_left_arm_joints)

                    # Omul mișcă mâna STÂNGĂ -> Calculează Pose Stânga -> Mută brațul DREPT al lui NAO
                    nao_right_arm_joints = jointsCalculator.left_hand_joints(pose_landmarker_result, human_left_hand_landmarks)
                    if nao_right_arm_joints is not None:
                        jointsCalculator.log_stability("R", nao_right_arm_joints)
                    
                    
                    # nao_head_joints = jointsCalculator.calculate_head_tracking(pose_landmarker_result.pose_landmarks[0])


                    timp_acum = time.time()
                    if timp_acum - LAST_COMMAND_TIME > COMMAND_INTERVAL:
                        

                        dispatch_time = time.time()
                        pipeline_latency_ms = (dispatch_time - frame_start) * 1000
                        latency_log.append(pipeline_latency_ms)
                        # Trimitem datele către brațele corecte ale robotului
                        if nao_left_arm_joints:
                            send_Larm_angles(nao_left_arm_joints)

                        if nao_right_arm_joints:
                            send_Rarm_angles(nao_right_arm_joints)

                        # if nao_head_joints:
                            # send_head_angles(nao_head_joints)
                        
                        LAST_COMMAND_TIME = timp_acum
                        
                except Exception as e:
                    print(f"Eroare calcul unghiuri: {e}\n\n")
                    # nao_head_joints = None
            else:
                pass
                # No person visible — scan until someone appears
                # nao_head_joints = jointsCalculator.scan_head()

            # if nao_head_joints:
                    # send_head_angles(nao_head_joints)

            # Afișare
            cv2.imshow("NAO Mirror Game", annotated_image_bgr)

            # Ieșire cu 'q'
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break 

    except Exception as e:
        print(f"Eroare: {e}")
    finally:
        if latency_log:
            report = {
                "total_frames": total_frames,
                "pose_detection_rate": pose_detected_frames / total_frames,
                "hand_detection_rate": hand_detected_frames / total_frames,
                "latency_ms": {
                    "mean": statistics.mean(latency_log),
                    "median": statistics.median(latency_log),
                    "p95": sorted(latency_log)[int(0.95 * len(latency_log))],
                    "max": max(latency_log)
                },
                "inference_ms": {
                    "mean": statistics.mean(inference_log)
                },
                "joint_stability": jointsCalculator.get_stability_report()
            }
            with open("metrics_report.json", "w") as f:
                json.dump(report, f, indent=2)
            print("Metrics saved to metrics_report.json")
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
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    # --- PRIORITY 4: Disable Nagle's Algorithm to send frames immediately ---
    srv.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1) 
    
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