# 1. Start from Ubuntu 16.04 (Xenial) on the Intel architecture
FROM ubuntu:16.04

# 2. Set the WORKDIR
WORKDIR /app

# 3. Install Python 2.7, pip, C++ dependencies, AND setuptools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python2.7 \
        python-pip \
        python-setuptools \
        # Dependencies for naoqi's C++ libs
        libstdc++6 \
        libgcc1 \
        libglib2.0-0 \
        libffi6 \
        # Dependencies for images (like libjpeg)
        libjpeg-turbo8 \
        libfreetype6 \
        \
        # --- ADD THESE LINES FOR OPENCV ---
        libsm6 \
        libxext6 \
        libxrender1 && \
        # ------------------------------------
        \
    # Clean up apt cache
    rm -rf /var/lib/apt/lists/*
# 4. Install Python libraries
# Upgrade pip and setuptools to the last versions that support Python 2.7
RUN pip install --no-cache-dir --upgrade "pip<21.0" "setuptools<45.0.0" && \
    pip install --no-cache-dir \
        numpy==1.16.6 \
        opencv-python==4.2.0.32 \
        flask \
        wheel

# 5. Copy the NAOqi SDK into the image
COPY pynaoqi-python2.7* /opt/pynaoqi

# 6. Set the environment paths
ENV PYTHONPATH=/opt/pynaoqi/lib/python2.7/site-packages \
    LD_LIBRARY_PATH=/opt/pynaoqi/lib

# 7. Copy your bridge script
COPY nao_command_bridge.py /app/nao_command_bridge.py

# 8. Tell the container to run your script
CMD ["python2.7", "nao_command_bridge.py"]