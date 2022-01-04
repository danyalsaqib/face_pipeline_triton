import cv2
import numpy as np
from skimage import transform as trans
import onnxruntime
import json
import os
import time

from attrdict import AttrDict

import tritonclient.grpc as grpcclient
import tritonclient.grpc.model_config_pb2 as mc
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
from tritonclient.utils import triton_to_np_dtype

def parse_model(model_metadata, model_config):
    """
    Check the configuration of a model to make sure it meets the
    requirements for an image classification network (as expected by
    this client)
    """

    input_metadata = model_metadata.inputs[0]
    input_config = model_config.input[0]
    output_metadata = model_metadata.outputs

    output_batch_dim = (model_config.max_batch_size > 0)
    non_one_cnt = 0
    
    input_batch_dim = (model_config.max_batch_size > 0)
    expected_input_dims = 3 + (1 if input_batch_dim else 0)
    if len(input_metadata.shape) != expected_input_dims:
        raise Exception(
            "expecting input to have {} dimensions, model '{}' input has {}".
            format(expected_input_dims, model_metadata.name,
                    len(input_metadata.shape)))

    if type(input_config.format) == str:
        FORMAT_ENUM_TO_INT = dict(mc.ModelInput.Format.items())
        input_config.format = FORMAT_ENUM_TO_INT[input_config.format]

    output_names = []
    for out_layer in output_metadata:
        output_names.append(out_layer.name)

    if input_config.format == mc.ModelInput.FORMAT_NHWC:
        h = input_metadata.shape[1 if input_batch_dim else 0]
        w = input_metadata.shape[2 if input_batch_dim else 1]
        c = input_metadata.shape[3 if input_batch_dim else 2]
    else:
        c = input_metadata.shape[1 if input_batch_dim else 0]
        h = input_metadata.shape[2 if input_batch_dim else 1]
        w = input_metadata.shape[3 if input_batch_dim else 2]

    return (model_config.max_batch_size, input_metadata.name,
        output_names, c, h, w, input_config.format,
        input_metadata.datatype)


def requestGenerator(batched_image_data, input_name, output_name, dtype, model_name, model_version):
    client = httpclient

    # Set the input data
    inputs = [client.InferInput(input_name, batched_image_data.shape, dtype)]
    #print("Datatype: ", dtype)
    #print("Batch Datatype: ", batched_image_data.dtype)
    inputs[0].set_data_from_numpy(batched_image_data)
    outputs = []
    for out_layer in output_name:
        outputs.append(client.InferRequestedOutput(out_layer))

    yield inputs, outputs, model_name, model_version


def convert_http_metadata_config(_metadata, _config):
    _model_metadata = AttrDict(_metadata)
    _model_config = AttrDict(_config)

    return _model_metadata, _model_config

def trt_infer(blob, model_name, model_version=''):

    # Recognition
    #result = session.run([output_name], {input_name: blob})
    triton_client = httpclient.InferenceServerClient(
                    'localhost:8000', verbose=False, concurrency=1)
    batch_size = len(blob)
    #print("Batch Size: ", batch_size)
    responses = []
    async_requests = []

    #model_name = 'densenet_onnx'
    #model_version = ''
    #classes = 512

    model_metadata = triton_client.get_model_metadata(
        model_name=model_name, model_version=model_version)

    model_config = triton_client.get_model_config(
        model_name=model_name, model_version=model_version)
    model_metadata, model_config = convert_http_metadata_config(
        model_metadata, model_config)

    max_batch_size, input_name, output_name, c, h, w, format, dtype = parse_model(
        model_metadata, model_config)

    #print("Output Names: ", output_name)
    #print("Data Type: ", dtype)


    image_idx = 0
    #input_filenames=[]
    repeated_image_data = []
    #filenames = []
    npdtype = triton_to_np_dtype(dtype)
    blob = blob.astype(npdtype)

    """
    for i in batch_dict:
        print("Filnames: ", batch_dict[i])
        filenames.append(batch_dict[i])
    """

    for idx in range(batch_size):
        #input_filenames.append(filenames[image_idx])
        #print("Image_IDX: ", image_idx)
        #print("image_data value: ", image_data[0])
        repeated_image_data.append(blob[image_idx])
        image_idx = (image_idx + 1) % len(blob)

    if max_batch_size > 0:
        batched_image_data = np.stack(repeated_image_data, axis=0)
    else:
        batched_image_data = repeated_image_data[0]

    #print("Batched Image Data: ", batched_image_data)

    sent_count = 0

    for inputs, outputs, model_name, model_version in requestGenerator(
        batched_image_data, input_name, output_name, dtype, model_name, model_version):
        sent_count += 1
        async_requests.append(
            triton_client.async_infer(
                model_name,
                inputs,
                request_id=str(sent_count),
                model_version=model_version,
                outputs=outputs))
    for async_request in async_requests:
        responses.append(async_request.get_result())

    #print("Responses Length: ", len(responses(output_name)))

    output_array = []
    for out_layer in output_name:
        for response in responses:
            this_id = response.get_response()["id"]
            #print("Request {}, batch size {}".format(this_id, batch_size))
            output_single = response.as_numpy(out_layer)
            #print("Output Array: ", output_array)
            #print("Output Array Shape: ", output_single.shape)
            #postprocess(response, output_name, batch_size, max_batch_size > 0)
            output_single = output_single.tolist()
            output_array.append(output_single)

    #print("Inference Successful")
    #print("Responses: ", responses)
    #print("Async Requests: ", async_requests)
    #print("Output List Length: ", len(output_array))
    return output_array
