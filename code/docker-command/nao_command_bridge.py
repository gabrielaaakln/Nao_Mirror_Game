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
HEAD_DEFAULT_ANGLES = [0.0, -0.16]

#the speed at which the robot moves
FRACTION_MAX_SPEED = 0.15 

LEFT_ARM_TARGET_JOINTS = ["LShoulderPitch",
                 "LShoulderRoll",
                 "LElbowRoll",
                 "LElbowYaw",
                 "LWristYaw",
                 "LHand"]


RIGHT_ARM_TARGET_JOINTS = ["RShoulderPitch",
                 "RShoulderRoll",
                 "RElbowYaw",
                 "RElbowRoll",
                 "RWristYaw",
                 "RHand"]

HEAD_TARGET_JOINTS = ["HeadYaw",
                "HeadPitch"]


WORDS_FOR_RECOGNISION = ["START", "STOP"]

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

EYELASHES = [
        "FaceLedRight7", "FaceLedLeft7",
        "FaceLedLeft6", "FaceLedRight6"
    ]

def restore_eyelashes():
    for led in EYELASHES:
        leds.fadeRGB(led, 0xe68bbe, 0.0)

current_eye_color = "white"
def blink_loop():
    global leds, current_eye_color

    top_leds = [
        "FaceLedLeft0", "FaceLedLeft1", "FaceLedLeft2",
        "FaceLedRight0", "FaceLedRight1", "FaceLedRight2"
    ]

    bottom_leds = [
        "FaceLedLeft5", "FaceLedLeft4", "FaceLedLeft3",
        "FaceLedRight5", "FaceLedRight4", "FaceLedRight3"
    ]

    while True:
        time.sleep(random.uniform(3.0, 6.0))

        if leds:
            try:
                color_to_restore = current_eye_color

                with led_lock:
                    for led in bottom_leds + top_leds:
                        leds.post.fadeRGB(led, 0x000000, 0.02)

                    for led in bottom_leds+top_leds:
                        leds.post.fadeRGB(led, color_to_restore, 0.08)
                    
                    restore_eyelashes()
                    time.sleep(0.2)

            except Exception as e:
                print("BLINK ERROR: {}".format(e))








# --- VOICE MONITORING THREAD ---
def voice_monitor_loop():
    global game_active, current_eye_color
    
    last_word_seen = ""  # track the last word we already acted on

    print("Voice monitor started. Listening for 'begin' or 'finish'...")
    
    while True:
        try:
            data = memory.getData("WordRecognized")
            
            if data and len(data) == 2:
                word = data[0]
                confidence = data[1]
                
                # Only act if this is a NEW word we haven't handled yet
                if confidence > 0.40 and word != last_word_seen:
                    last_word_seen = word  # mark it as handled
                    
                    if word == WORDS_FOR_RECOGNISION[0] and not game_active:
                        if motion:
                            motion.setBreathEnabled("Body", False)
                        current_eye_color = "green"
                        with led_lock:
                            leds.fadeRGB("FaceLeds", "green", 0.2)
                        time.sleep(1)
                        game_active = True
                        current_eye_color = "white"
                        with led_lock:
                            leds.fadeRGB("FaceLeds", "white", 0.5)
                            restore_eyelashes()

                    elif word == WORDS_FOR_RECOGNISION[1] and game_active:
                        motion.post.setAngles(LEFT_ARM_TARGET_JOINTS, L_HAND_DEFAULT_ANGLES, FRACTION_MAX_SPEED)
                        motion.post.setAngles(RIGHT_ARM_TARGET_JOINTS, R_HAND_DEFAULT_ANGLES, FRACTION_MAX_SPEED)
                        motion.post.setAngles(HEAD_TARGET_JOINTS, HEAD_DEFAULT_ANGLES, FRACTION_MAX_SPEED)
                        idle_robot()
                        current_eye_color = "red"
                        with led_lock:
                            leds.fadeRGB("FaceLeds", "red", 0.2)
                        time.sleep(1)
                        game_active = False
                        current_eye_color = "white"
                        with led_lock:
                            leds.fadeRGB("FaceLeds", "white", 0.5)
                            restore_eyelashes()

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

        print("Setting Autonomous Life to disabled...")
        life.setState("disabled")

        print("Waking up robot...")
        motion.wakeUp()

        leds.setIntensity("FaceLeds", 1.0)
        motion.post.setAngles("HeadPitch", 0.1, FRACTION_MAX_SPEED)
        # idle_robot()
        
        # --- SETUP SPEECH RECOGNITION ---
        print("Setting up Speech Recognition...")
        try:
            speech.unsubscribe("MirrorGameVoice")
        except:
            pass
        
        speech.pause(True)
        speech.setLanguage("English")
        speech.setWordListAsVocabulary(WORDS_FOR_RECOGNISION)
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


def idle_robot():
    global motion, leds

    if leds:
            try:
                with led_lock:
                    for led in EYELASHES:
                        leds.fadeRGB(led, 0xe68bbe, 0.0)
            except:
                print("The eyelashes couldnt work!")
    motion.setBreathEnabled("Body", True)


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
            
            if command.get("type") == "joints" and "LArmJoints" in command:
                joints = command["LArmJoints"]
                
                names = []
                values = []
                
                for j_name in LEFT_ARM_TARGET_JOINTS:
                    if j_name in joints:
                        names.append(str(j_name))           
                        values.append(float(joints[j_name])) 
                
                if names:
                    motion.post.setAngles(names, values, FRACTION_MAX_SPEED)


            if command.get("type") == "joints" and "RArmJoints" in command:
                joints = command["RArmJoints"]
                
                names = []
                values = []
                
                for j_name in RIGHT_ARM_TARGET_JOINTS:
                    if j_name in joints:
                        names.append(str(j_name))           
                        values.append(float(joints[j_name])) 
                
                if names:
                    motion.post.setAngles(names, values, FRACTION_MAX_SPEED)


            if command.get("type") == "joints" and "HeadJoints" in command:
                joints = command["HeadJoints"]
                
                names = []
                values = []
                
                for j_name in HEAD_TARGET_JOINTS:
                    if j_name in joints:
                        names.append(str(j_name))           
                        values.append(float(joints[j_name])) 
                
                if names:
                    motion.post.setAngles(names, values, 0.07)

        except socket.timeout:
            # --- TIMEOUT LOGIC ---
            # No data received for 0.5 seconds. Check if we should reset.
            if game_active and not is_in_default_position:
                if time.time() - last_packet_time > 1.5: # 1.5 seconds without commands
                    print("\n[!] Connection lost/paused. Returning to default position.")
                    motion.post.setAngles(LEFT_ARM_TARGET_JOINTS, L_HAND_DEFAULT_ANGLES, FRACTION_MAX_SPEED)
                    motion.post.setAngles(RIGHT_ARM_TARGET_JOINTS, R_HAND_DEFAULT_ANGLES, FRACTION_MAX_SPEED)
                    motion.post.setAngles(HEAD_TARGET_JOINTS, HEAD_DEFAULT_ANGLES, 0.07)
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