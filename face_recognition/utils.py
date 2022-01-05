import cv2
import numpy as np
from skimage import transform as trans
import onnxruntime
import json
import os
from mtcnn_ort import MTCNN
import time
from trt_inference import trt_infer, trt_cosineDistance

src = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041]], dtype=np.float32)
src[:, 0] += 8.0

tform = trans.SimilarityTransform()
distance_threshold = 0.55

roll_threshold = 80 #20
yaw_threshold = 45 # 35

# Reading dictionary
f = open('embeddings.json')
data = json.load(f)
print("Dict length : ", len(data["embeddings"]))

def detection_results_show(image_path, faces):

    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    for i in range(len(faces[0])):
        cv2.rectangle(image,
                    (int(faces[0][i][0]), int(faces[0][i][1])),
                    (int(faces[0][i][2]), int(faces[0][i][3])),
                    (0,155,255), 2)

        cv2.circle(image,(int(faces[1][0][i]), int(faces[1][5][i])), 2, (0,155,255), 2)
        cv2.circle(image,(int(faces[1][1][i]), int(faces[1][6][i])), 2, (0,155,255), 2)
        cv2.circle(image,(int(faces[1][2][i]), int(faces[1][7][i])), 2, (0,155,255), 2)
        cv2.circle(image,(int(faces[1][3][i]), int(faces[1][8][i])), 2, (0,155,255), 2)
        cv2.circle(image,(int(faces[1][4][i]), int(faces[1][9][i])), 2, (0,155,255), 2)

    cv2.imshow("Input", image)
    cv2.waitKey(0)

def cropped_results_show(image, faces):

    cv2.circle(image,(int(faces[0]), int(faces[5])), 2, (0,155,255), 2)
    cv2.circle(image,(int(faces[1]), int(faces[6])), 2, (0,155,255), 2)
    cv2.circle(image,(int(faces[2]), int(faces[7])), 2, (0,155,255), 2)
    cv2.circle(image,(int(faces[3]), int(faces[8])), 2, (0,155,255), 2)
    cv2.circle(image,(int(faces[4]), int(faces[9])), 2, (0,155,255), 2)

    cv2.imshow("cropped Image", image)
    cv2.waitKey(0)

def cropped_results(image):

    cv2.imshow("Warped Image", image)
    cv2.waitKey(0)

def find_roll(pts):
    return pts[6] - pts[5]

def find_yaw(pts):
    le2n = pts[2] - pts[0]
    re2n = pts[1] - pts[2]
    if le2n < 0 or re2n < 0:
        return 100
    return le2n - re2n

def find_pitch(pts):
    eye_y = (pts[5] + pts[6]) / 2
    mou_y = (pts[8] + pts[9]) / 2
    e2n = eye_y - pts[7]
    n2m = pts[7] - mou_y
    return e2n/n2m

def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def infer_image(img_path, detector):

    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    #image = cv2.resize(image, (1280, 720), interpolation = cv2.INTER_AREA)
    start = time.time()
    faces = detector.detect_faces_raw(image)
    end = time.time()
    #print("Detection Time = ", end - start)
    celeb = []
    #detection_results_show(img_path, faces)

    for i in range(len(faces[0])):
        # Size of Image is less than 100 pixel,scrap it
        image = cv2.imread(img_path)
        if ((int(faces[0][i][3]) - int(faces[0][i][1])) > 100 or (int(faces[0][i][2]) - int(faces[0][i][0])) > 100):

            crop_image = image[int(faces[0][i][1]): int(faces[0][i][3]), int(faces[0][i][0]):int(faces[0][i][2])]
            landmarks = [int(faces[1][0][i]), int(faces[1][1][i]), int(faces[1][2][i]), int(faces[1][3][i]), int(faces[1][4][i]),
                        int(faces[1][5][i]), int(faces[1][6][i]), int(faces[1][7][i]), int(faces[1][8][i]), int(faces[1][9][i])]
            
            #print("Roll : ", find_roll(landmarks))
            #print("Yaw : ", find_yaw(landmarks))
            #print("Pitch : ", find_pitch(landmarks))
            #cropped_results_show(crop_image, landmarks)
            
            if find_roll(landmarks) > - roll_threshold and  find_roll(landmarks) < roll_threshold and find_yaw(landmarks) > -yaw_threshold and find_yaw(landmarks) < yaw_threshold and find_pitch(landmarks) < 2 and find_pitch(landmarks) > 0.5:
                
                # Face Warping
                facial5points = np.reshape(landmarks, (2, 5)).T
                tform.estimate(facial5points, src)
                M = tform.params[0:2, :]
                img = cv2.warpAffine(image, M, (112, 112), borderValue=0.0)
                #cropped_results(img)
                
                # Recognition Preprocessing
                blob = cv2.dnn.blobFromImage(img, 1, (112, 112), (0, 0, 0))
                blob -= 127.5
                blob /= 128
                
                # Recognition
                #result = session.run([output_name], {input_name: blob})
                #print("Result Shape ONNX: ", len(result[0][0]))
                result = trt_infer(blob, model_name='arcface_onnx')
                print("Result Shape Triton ArcFace: ", len(result[0][0]))

                # Matching with the existing embeddings
                distance = 1000
                best_distance = 1
                best_index = -1
                #start = time.time()
                for j in range(len(data["embeddings"])):
                    #distance = findCosineDistance(result[0][0], data["embeddings"][j])
                    distance = trt_cosineDistance(result[0][0], data["embeddings"][j])
                    if (distance < distance_threshold) and (distance < best_distance):
                        best_index = j
                        best_distance = distance
                #end = time.time()
                #print("Embedding time : ", end - start)

                if (best_index != -1):
                    celeb += [data["names"][best_index]]
                    #print("Celeb : ", data["names"][best_index])
        

            else:
                #print("Roll : ", find_roll(landmarks))
                #print("Yaw : ", find_yaw(landmarks))
                #print("Pitch : ", find_pitch(landmarks))
                print("Invalid Pose")
        else:
            #pass
            print("Image size is small")
    return celeb

def check_generate_embedding(img_path, detector, session, input_name, output_name):

    if not (os. path.exists(img_path)):
        return "Inavlid Image Path"
        
    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces_raw(image)
    celeb = []
    #detection_results_show(img_path, faces)
    if len(faces[0]) != 1:
        if len(faces[0]) < 1:
            return "No face detected"
        elif len(faces[0]) > 1:
            return "Multiple face detected"

    # Size of Image is less than 100 pixel,scrap it
    image = cv2.imread(img_path)
    if ((int(faces[0][0][3]) - int(faces[0][0][1])) > 100 or (int(faces[0][0][2]) - int(faces[0][0][0])) > 100):

        crop_image = image[int(faces[0][0][1]): int(faces[0][0][3]), int(faces[0][0][0]):int(faces[0][0][2])]
        landmarks = [int(faces[1][0][0]), int(faces[1][1][0]), int(faces[1][2][0]), int(faces[1][3][0]), int(faces[1][4][0]),
                    int(faces[1][5][0]), int(faces[1][6][0]), int(faces[1][7][0]), int(faces[1][8][0]), int(faces[1][9][0])]
        
        if find_roll(landmarks) > - 20 and  find_roll(landmarks) < 20 and find_yaw(landmarks) > -35 and find_yaw(landmarks) < 35 and find_pitch(landmarks) < 2 and find_pitch(landmarks) > 0.5:
            
            # Face Warping
            facial5points = np.reshape(landmarks, (2, 5)).T
            tform.estimate(facial5points, src)
            M = tform.params[0:2, :]
            img = cv2.warpAffine(image, M, (112, 112), borderValue=0.0)
            #cropped_results(img)
            
            # Recognition Preprocessing
            blob = cv2.dnn.blobFromImage(img, 1, (112, 112), (0, 0, 0))
            blob -= 127.5
            blob /= 128
            result = session.run([output_name], {input_name: blob})                

        else:
            return "Invalid Pose"
    else:
        return "Image size is small"
    return result[0][0]

def find_element_in_list(element, list_element):
    try:
        index_element = list_element.index(element)
        return index_element
    except ValueError:
        return None
