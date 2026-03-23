from __future__ import print_function
from flask import Flask, request, jsonify
from naoqi import ALProxy
import sys
import time
import signal

app = Flask(__name__)

#Configurare
# ROBOT_IP = "10.195.28.34" //flowerpower
ROBOT_IP = "172.20.10.2" #gabitzu
ROBOT_PORT = 9559

# Variabile globale
motion = None
life = None
tts = None

# --- LIFECYCLE FUNCTIONS ---

def initialize_robot():
    """Connects to the robot, handles Autonomous Life, and Wakes up."""
    global motion, life, tts
    try:
        print("Connecting to NAO at {}:{}...".format(ROBOT_IP, ROBOT_PORT))
        motion = ALProxy("ALMotion", ROBOT_IP, ROBOT_PORT)
        life = ALProxy("ALAutonomousLife", ROBOT_IP, ROBOT_PORT)
        tts = ALProxy("ALTextToSpeech", ROBOT_IP, ROBOT_PORT)

        # Disable Autonomous Life so it doesn't fight your poses
        if life.getState() != "disabled":
            print("Disabling Autonomous Life...")
            life.setState("disabled")

        # Wake Up (Stiffness 1.0 + Stand Init)
        print("Waking up robot...")
        motion.wakeUp()
        motion.setStiffnesses("RArm", 1.0)
        motion.setAngles("HeadPitch", 0.1, 0.2)
        print("Robot is Awake and Ready.")
        
    except Exception as e:
        print("CRITICAL ERROR: Could not connect to NAO.", e)
        sys.exit(1)

def shutdown_robot(signum=None, frame=None):
    """Safely relaxes the robot on server exit."""
    global motion
    print("\nShutting down... Resting robot.")
    if motion:
        try:
            motion.rest()
        except Exception as e:
            print("Error during rest:", e)
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown_robot)
signal.signal(signal.SIGTERM, shutdown_robot)


# --- FLASK ENDPOINT ---

@app.route('/movement', methods=['POST'])
def update_movement():
    global current_command, last_update_time
    
    data = request.json
    
    if data:
        current_command = data
        last_update_time = time.time()
        
        # --- UPDATED DEBUG LOGGING ---
        # The new client sends: {"type": "angles", "joints": {...}}
        if data.get("type") == "joints" and "joints" in data:
            joints = data["joints"]
            
            # Cast unicode keys to standard strings
            names = [str(k) for k in joints.keys()]
            
            # Ensure angles are floats
            angles = [float(v) for v in joints.values()]

            fractionMaxSpeed = 0.1

            motion.setAngles(names, angles, fractionMaxSpeed)
            # # Print specific joint for debugging if it exists
            # if "LShoulderRoll" in joints:
            #     # print(f"SET: RShoulderRoll = {joints['RShoulderRoll']}")
            #     motion.setAngles("LShoulderRoll", joints["LShoulderRoll"], 0.2)
            # else:
            #     print("LShouolderRoll failed!!")
        
            # # Fallback for old position data or other types
            # if "LShoulderPitch" in joints:
            #     # print(f"SET: RShoulderPitch = {joints['RShoulderPitch']}")
            #     motion.setAngles("LShoulderPitch", joints["LShoulderPitch"], 0.2)
            # else:
            #     print("LShoulderPitch failed!!")

            # if "LElbowRoll" in joints:
            #     motion.setAngles("LElbowRoll", joints["LElbowRoll"], 0.2)
            # else:
            #     print("LElbowRoll failed!!")

            # if "LElbowYaw" in joints:
            #     motion.setAngles("LElbowYaw", joints["LElbowYaw"], 0.2)
            # else:
            #     print("LElbowYaw failed!!")
        
    return jsonify({"status": "success"}), 200

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "alive", "robot_ip": ROBOT_IP}), 200

if __name__ == "__main__":
    initialize_robot()
    print("Starting Server on 0.0.0.0:5050")
    try:
        app.run(host="0.0.0.0", port=5050)
    except Exception as e:
        print("Server crashed:", e)
    finally:
        shutdown_robot()