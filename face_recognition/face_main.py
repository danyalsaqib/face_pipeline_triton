import cv2
import numpy as np
from utils import *
import time
import json

# Initializing the detector
detector = MTCNN()

# Initializing the recognizer
#model_path = "models/arcface.onnx"
#session = onnxruntime.InferenceSession(model_path, None)
#input_name = session.get_inputs()[0].name
#output_name = session.get_outputs()[0].name

file_batch_dict = {
    "frame_2" : "images/11.png"
}


# Returns the stored embeddings length
def get_embeddings_count():
    f = open('embeddings.json')
    data = json.load(f)
    f.close()
    return len(data["embeddings"])


# Returns the names of stored embeddings
def get_embeddings_name():
    f = open('embeddings.json')
    data = json.load(f)
    f.close()
    return data["names"]


# Adds the embedding to the embeddings.json 
def add_embedding(image_path, name):
    embedding = check_generate_embedding(image_path, detector, session, input_name, output_name)

    if isinstance(embedding, (np.ndarray, np.generic) ):
        with open("embeddings.json", "r") as jsonFile:
            data = json.load(jsonFile)

        data["names"].append(name)
        #print(data["names"])
        data["embeddings"].append(embedding.tolist())

        with open("embeddings.json", "w") as jsonFile:
            json.dump(data, jsonFile)
    else: 
        print(embedding)


# Delete the embedding from the embeddings.json
def delete_embedding(name): 
    with open("embeddings.json", "r") as jsonFile:
        data = json.load(jsonFile)

    index = find_element_in_list(name, data["names"])

    if index != None:
        data["embeddings"].pop(index)
        data["names"].pop(index)
        print("Deleted embeding of ", name)
    else:
        print("Didn't find the embedding with name ", name)

    with open("embeddings.json", "w") as jsonFile:
        json.dump(data, jsonFile)


# Returns the results of inference
def infer_batch(batch_dict):
    results = {}
    start = time.time()
    for image_path in batch_dict:
        #print("Frame : ", batch_dict[image_path])
        results[image_path] = infer_image(batch_dict[image_path], detector)
    end = time.time()
    #print("Total time : ", end - start)

    print(results)
    return results


if __name__ == '__main__':
    infer_batch(file_batch_dict)
    #for filename in os.listdir("Pictures"):
    #    name = os.path.splitext(filename)[0]
    #    img_path = os.path.join("Pictures",filename)
    #    print(name)
    #    add_embedding(img_path, name)
