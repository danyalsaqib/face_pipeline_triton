# face_pipeline_triton
FRS Pipeline for media_v1, implemented on triton.

## Downloading the ArcFace and MTCNN Models
Run the script 'face_recognition/models/get_model.py' using the following commands to automatically download the model:
```
python3 face_recognition/models/get_model.py
```

## Dockerfile
The dockerfile contains instructions to first install necessary libraries, and then copy the 'retinaface_functions/' directory into a local docker directory using the command `COPY face_recognition /face_recognition`. The docker uses the tensorflow container as its parent, given by `FROM nvcr.io/nvidia/tensorflow:21.11-tf2-py3`

To build the docker, simply use the following command. Remember that the dot at the end is important:
```
sudo docker build -t face_pipeline_v1:single .
```
To run the docker after successfully building it, use the following command:
```
sudo docker run --gpus all -it --rm face_pipeline_v1:single
```

## Running the complete pipeline
After running the docker, change into the following directory:
```
cd /face_recognition/
````

The 'python3 face_main.py/face_main.py' file has a main functon, that can run the entire pipeline with 1 call. Its usage is as follows:
```
python3 face_main.py
```

As an example, by default, the inference runs on a sample image. The inference is essentially done by calling the infer_image function, and passing it a path to an image. As an example, it runs:
```
test_pic = "image_1.jpeg"
infer_image(test_pic)
```
The 'face_infer_batch' takes a dictionary containing names, and respective paths of images. It returns a dictionary, containing the recognized face in each image. An example code is something like this:
```
if __name__ == '__main__':
    
    file_batch_dict = {
      "frame_1" : "images/image_1.jpeg",
      "frame_2" : "images/image_2.jpeg",
      "frame_3" : "images/image_3.jpeg",
      "frame_4" : "images/image_4.jpeg",
      "frame_5" : "images/image_5.jpeg"
    }
    
    face_infer_batch(file_batch_dict)
```
