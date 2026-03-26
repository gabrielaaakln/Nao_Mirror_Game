from __future__ import print_function
from naoqi import ALProxy
import socket
import json
import sys
import time
import random
import os
import threading

# --- Configuration ---
ROBOT_IP = "172.20.10.4"
ROBOT_PORT = 9559
UDP_IP = "0.0.0.0"
UDP_PORT = 5050

#for reinitialising the robot
R_HAND_DEFAULT_ANGLES = [1.44, -0.22, 1.19, 0.40, 0.1, 0.30]
L_HAND_DEFAULT_ANGLES = [1.47, 0.19, -0.40, -1.18, 0.09, 0.30]

#the speed at which the robot moves
FRACTION_MAX_SPEED = 0.15 

TARGET_JOINTS = ["LShoulderPitch",
                 "LShoulderRoll",
                 "LElbowRoll",
                 "LElbowYaw",
                 "LWristYaw",
                 "LHand"]

# Global variables
motion = None
life = None
tts = None
leds = None
speech = None
memory = None


# --- THREAD SAFETY LOCK ---
# This prevents the socket from crashing if two threads try to change LEDs at the exact same time!
led_lock = threading.Lock()

# Game State Flag (Start as False so it waits for "START")
game_active = False

# Blinking script
current_eye_color = "blue"

def blink_loop():
    """Background thread to make NAO blink naturally."""
    global leds, current_eye_color
    
    while True:
        # Wait a random amount of time between blinks (3 to 6 seconds)
        time.sleep(random.uniform(3.0, 6.0))
        
        if leds:
            try:
                # Save the current color so we know what to return to
                color_to_restore = current_eye_color
                
                # SAFELY lock the proxy and blink
                with led_lock:
                    leds.fadeRGB("FaceLeds", "black", 0.1)
                    time.sleep(0.1)
                    leds.fadeRGB("FaceLeds", color_to_restore, 0.1)
            except Exception as e:
                pass # Fail silently so a tiny blink error doesn't spam the console


# --- VOICE MONITORING THREAD ---
def voice_monitor_loop():
    """Background thread that constantly checks if NAO heard a command."""
    global game_active, current_eye_color
    
    print("Voice monitor started. Listening for 'start' or 'stop'...")
    
    while True:
        try:
            data = memory.getData("WordRecognized")
            
            if data and len(data) == 2:
                word = data[0]
                confidence = data[1]
                
                if confidence > 0.4:
                    if word == "start" and not game_active:
                        print("\n>>> VOICE COMMAND: 'START'. GAME ON! Eyes turning GREEN.")
                        current_eye_color = "green"
                        with led_lock:
                            leds.fadeRGB("FaceLeds", "green", 0.2)
                        game_active = True
                        
                    elif word == "stop" and game_active:
                        print("\n>>> VOICE COMMAND: 'STOP'. GAME PAUSED. Eyes turning BLUE again.")
                        current_eye_color = "blue"
                        with led_lock:
                            leds.fadeRGB("FaceLeds", "blue", 0.5)
                        game_active = False
                        motion.post.setAngles(TARGET_JOINTS, L_HAND_DEFAULT_ANGLES, FRACTION_MAX_SPEED)

        except Exception as e:
            pass 
            
        time.sleep(0.5) 


# --- LIFECYCLE FUNCTIONS ---

def initialize_robot():
    """Connects to the robot, handles Autonomous Life, and Wakes up."""
    global motion, life, tts, leds, speech, memory
    try:
        print("Connecting to NAO at {}:{}...".format(ROBOT_IP, ROBOT_PORT))
        motion = ALProxy("ALMotion", ROBOT_IP, ROBOT_PORT)
        life = ALProxy("ALAutonomousLife", ROBOT_IP, ROBOT_PORT)
        tts = ALProxy("ALTextToSpeech", ROBOT_IP, ROBOT_PORT)
        leds = ALProxy("ALLeds", ROBOT_IP, ROBOT_PORT)
        speech = ALProxy("ALSpeechRecognition", ROBOT_IP, ROBOT_PORT)
        memory = ALProxy("ALMemory", ROBOT_IP, ROBOT_PORT)

        if life.getState() != "disabled":
            print("Disabling Autonomous Life...")
            life.setState("disabled")

        print("Waking up robot...")
        motion.wakeUp()
        motion.setStiffnesses("RArm", 1.0)
        motion.setAngles("HeadPitch", 0.1, 0.2)

        # --- INITIALIZE GAME STATE ---
        print("Setting initial state (Blue Eyes)...")
        with led_lock:
            leds.fadeRGB("FaceLeds", "blue", 0.5)
        
        # --- SETUP SPEECH RECOGNITION ---
        print("Setting up Speech Recognition...")
        try:
            speech.unsubscribe("MirrorGameVoice")
        except:
            pass
        
        speech.pause(True)
        speech.setLanguage("English")
        speech.setVocabulary(["start", "stop"], False)
        speech.pause(False)
        speech.subscribe("MirrorGameVoice")
        
        # Start Threads
        monitor_thread = threading.Thread(target=voice_monitor_loop)
        monitor_thread.daemon = True
        monitor_thread.start()

        blink_thread = threading.Thread(target=blink_loop)
        blink_thread.daemon = True
        blink_thread.start()

        print("Robot is Awake and Ready.")
    except Exception as e:
        print("CRITICAL ERROR: Could not connect to NAO.", e)
        sys.exit(1)


def shutdown_robot(signum=None, frame=None):
    """Safely relaxes the robot on server exit."""
    global motion, speech, leds

    print("\nShutting down... Resting robot.")

    if leds:
        try:
            with led_lock:
                leds.fadeRGB("FaceLeds", "white", 0.5)
            print("Leds turning white")
        except:
            pass

    if speech:
        try:
            speech.unsubscribe("MirrorGameVoice")
            print("Microphone unsubscribed safely.")
        except:
            pass
    
    if motion:
        try:
            motion.rest()
        except Exception as e:
            pass
            
    os._exit(0)


# --- UDP SERVER ENDPOINT ---

def run_udp_server():
    """Listens for rapid UDP packets and applies them to the robot."""
    global motion, game_active
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    
    # --- ADD A TIMEOUT (0.5 seconds) ---
    sock.settimeout(0.5) 
    print("Listening for UDP commands on {}:{}".format(UDP_IP, UDP_PORT))

    last_packet_time = time.time()
    is_in_default_position = True # Assume it starts in default

    while True:
        try:
            data, addr = sock.recvfrom(1024) 
            
            # We got a packet! Update the timer and flag.
            last_packet_time = time.time()
            is_in_default_position = False
            
            if not game_active:
                continue

            command = json.loads(data.decode('utf-8'))
            
            if command.get("type") == "joints" and "joints" in command:
                joints = command["joints"]
                
                names = []
                values = []
                
                for j_name in TARGET_JOINTS:
                    if j_name in joints:
                        names.append(str(j_name))           
                        values.append(float(joints[j_name])) 
                
                if names:
                    motion.post.setAngles(names, values, FRACTION_MAX_SPEED)

        except socket.timeout:
            # --- TIMEOUT LOGIC ---
            # No data received for 0.5 seconds. Check if we should reset.
            if game_active and not is_in_default_position:
                if time.time() - last_packet_time > 1.5: # 1.5 seconds without commands
                    print("\n[!] Connection lost/paused. Returning to default position.")
                    motion.post.setAngles(TARGET_JOINTS, L_HAND_DEFAULT_ANGLES, FRACTION_MAX_SPEED)
                    is_in_default_position = True

        except KeyboardInterrupt:
            break 
        except ValueError:
            pass # Ignore JSON errors
        except Exception as e:
            print("UDP Server error:", e)



if __name__ == "__main__":
    initialize_robot()
    try:
        run_udp_server()
    except KeyboardInterrupt:
        print("\nCTRL+C detected!")
    except Exception as e:
        print("Server crashed:", e)
    finally:
        shutdown_robot()