# Mirror Game with NAO Robot

**Team:** Meowtrix (Dascalu Ioana-Felicia & Diaconu Gabriel) <br>
**Institution:** “Gheorghe Asachi” Technical University of Iasi - Faculty of Automatic Control and Computer Engineering <br>
**Context:** Project for the Image Processing course, 2025

---

## Project Description

This project implements a **"Mirror Game"** type system, with the objective of developing a complete software solution that allows the NAO humanoid robot to imitate a human user’s gestures in real time.

The solution uses **Computer Vision** techniques (through Google MediaPipe) to detect human body keypoints (landmarks) and translate these movements into specific motor commands for the NAO robot.

### Main Functionalities

The system is capable of recognizing and executing the following interactions:

1. **Wave:** The robot detects a waving motion and responds verbally ("Hello there, buddy!") and physically.
2. **Peace Sign:** The robot recognizes the peace sign and responds accordingly ("Peace to you too, my friend!").
3. **Power Pose:** The robot detects arm flexing (biceps) and replicates the gesture enthusiastically.

---

## System Architecture

The architecture is distributed and modular, using Docker containers to manage the robot’s legacy dependencies (Python 2.7) and a modern environment for image processing.

### Components:

1. **NAO Camera Bridge (Docker Container)**

   * Runs on Python 2.7 using the `naoqi` SDK.
   * Captures the video stream from the robot’s camera and transmits it via TCP socket to the processing unit.
   * Script: `nao_camera_bridge.py`.

2. **Image Processing Unit (Laptop/PC)**

   * Python 3 application that uses **MediaPipe** for skeleton and hand detection.
   * Analyzes gesture geometry and, upon validating a movement, sends an HTTP POST request to the robot.
   * Script: `image_processing.py`.

3. **NAO Command Bridge (Flask Microservice)**

   * Flask server running on Python 2.7 with `naoqi`.
   * Receives commands from the PC and controls the robot’s motors to execute movements.
   * Script: `nao_command_bridge.py`.

**Data flow:** `NAO Camera` -> `Docker Bridge` -> `TCP` -> `MediaPipe Processing` -> `HTTP` -> `Flask Server` -> `NAO Motors`.

---

## Installation and Execution Guide

To replicate this project, you need a NAO robot, the `pynaoqi` SDK, and Docker installed on your computer.

### Step 1: IP Configuration

Before running, IP addresses must be updated in the scripts to match the local network:

* In `nao_camera_bridge.py`: Set `NAO_IP` (robot IP) and `MAC_RECEIVER_IP` (PC IP).
* In `nao_command_bridge.py`: Set `ROBOT_IP`.
* In `image_processing.py`: Check the `NAO_BRIDGE_URL` variable.

### Step 2: Preparing the Docker Image

Since the NAOqi SDK requires specific dependencies (Ubuntu 16.04), Docker is used. Make sure the SDK archive (`pynaoqi-python2.7...`) is located in the project folder.

```bash
# Build Docker images
# * from the specific folder of each Docker image, run separately, in separate terminals for easier execution:
docker build --platform linux/amd64 -t nao_video_bridge:py2 .
docker build --platform linux/amd64 -t nao_command_bridge:py2 .
```

### Step 3: Running the Video Bridge

This container starts the script that streams images from the robot to the PC.

```bash
# Run with host network for direct network access
docker run nao_video_bridge:py2  
```

### Step 4: Running Image Processing (immediately after starting the Video Bridge)

This is the main application ("the brain") running on the host machine with Python 3.

```bash
# Install a Python 3.11 version (RECOMMENDED: 3.11.14)
# Create a virtual environment and activate it 

# Install dependencies
pip install -r requirements.txt

# Run application
python image_processing.py
```

### Step 5: Starting the Command Server

This service also runs via Docker. It listens on port 5050.

```bash
docker run -p 5050:5050/udp nao_command_bridge:py2
```

### Step 6: Shutdown

```bash
# To stop, press the 'q' key
# The script responsible for transmitting commands must be closed from the terminal using 'CTRL+C'
```

---

### Performance and Limitations

* Accuracy: High success rate (up to 100% under optimal testing conditions) for predefined gestures.
* Latency: Average of approximately 128ms between detection and command.
* Limitations: The low resolution of the NAO camera and the limited agility of the robot’s joints can affect imitation fluidity.

---

### References

1. Aldebaran Documentation - NAOqi Motion
2. Google AI Edge - MediaPipe Solutions
3. Flask Documentation
4. Dockerize your Flask App - GeeksForGeeks
