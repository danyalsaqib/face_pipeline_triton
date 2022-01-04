FROM ubuntu

ARG DEBIAN_FRONTEND=noninteractive

COPY requirements.txt ./

RUN apt-get update -y && \
    apt-get install -y python3.6 python3-distutils python3-pip python3-apt && \
    python3 -m pip install --upgrade pip && \
    apt-get install ffmpeg libsm6 libxext6  -y && \
    pip install -r requirements.txt
    
COPY face_recognition /face_recognition
#CMD ["python3", "/face_recognition/face_main.py"]
#CMD ["sh","-c","python3 /face_recognition/models/get_model.py && python3 /face_recognition/face_main.py"]
