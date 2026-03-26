# nao_camera_bridge.py
from naoqi import ALProxy
import socket
import struct
import cv2
import numpy as np
import sys
import time

NAO_IP = "172.20.10.4"         
NAO_PORT = 9559
MAC_RECEIVER_IP = "172.20.10.5"
MAC_RECEIVER_PORT = 5001

def connect_video():
    video = ALProxy("ALVideoDevice", NAO_IP, NAO_PORT)
    # subscribeCamera(clientName, cameraIndex, resolution, colorSpace, fps)
    # cameraIndex: 0=top, 1=bottom; resolution: 0=QQVGA, 1=QVGA; colorSpace: 11=RGB, fps=30
    sub = video.subscribeCamera("nao_bridge", 0, 1, 13, 30)
    return video, sub

def get_image_array(nao_img):
    # nao_img is a tuple: [width, height, numLayers, timestamp, left, top, dataString]
    width = nao_img[0]
    height = nao_img[1]
    # data is bytes string in RGB order
    data = nao_img[6]
    arr = np.frombuffer(data, dtype=np.uint8)
    try:
        arr = arr.reshape((height, width, 3))
    except Exception as e:
        # fallback: sometimes buffer may be contiguous differently
        print("reshape error:", e)
        return None
    return arr

def send_jpeg_over_tcp(sock, jpeg_bytes):
    # prefix with 4-byte length in network byte order
    length = struct.pack("!I", len(jpeg_bytes))
    sock.sendall(length + jpeg_bytes)

def main():
    print("Connecting to NAO at", NAO_IP)
    video, sub = connect_video()

    print("Connecting to receiver at {}:{}".format(MAC_RECEIVER_IP, MAC_RECEIVER_PORT))
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # --- PRIORITY 4: Disable Nagle's Algorithm ---
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    # [Keep your connection retry loop]
    # Attempt multiple times if network not ready
    for i in range(10):
        try:
            sock.connect((MAC_RECEIVER_IP, MAC_RECEIVER_PORT))
            break
        except Exception as e:
            print("Connect attempt", i, "failed:", e)
            time.sleep(1)
    else:
        print("Could not connect to receiver. Exiting.")
        sys.exit(1)

    try:
        while True:
            img = video.getImageRemote(sub)
            if img is None:
                continue
            arr = get_image_array(img)
            if arr is None:
                continue

            # --- PRIORITY 5: Removed cv2.cvtColor! We encode 'arr' directly since it's already BGR ---
            ret, jpg = cv2.imencode(".jpg", arr)

            if not ret:
                continue
            send_jpeg_over_tcp(sock, jpg.tobytes())
    
    except KeyboardInterrupt:
        pass
    finally:
        try:
            video.unsubscribe(sub)
        except:
            pass
        sock.close()



if __name__ == "__main__":
    main()