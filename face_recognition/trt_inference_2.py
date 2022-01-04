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
    #if len(model_metadata.inputs) != 1:
    #    raise Exception("expecting 1 input, got {}".format(
    #        len(model_metadata.inputs)))
    #if len(model_metadata.outputs) != 1:
    #    raise Exception("expecting 1 output, got {}".format(
    #        len(model_metadata.outputs)))

    #if len(model_config.input) != 1:
    #    raise Exception(
    #        "expecting 1 input in model configuration, got {}".format(
    #            len(model_config.input)))

    input_metadata = model_metadata.inputs[0]
    input_config = model_config.input[0]
    output_metadata = model_metadata.outputs

    #if output_metadata.datatype != "FP32":
    #    raise Exception("expecting output datatype to be FP32, model '" +
    #                    model_metadata.name + "' output type is " +
    #                    output_metadata.datatype)

    # Output is expected to be a vector. But allow any number of
    # dimensions as long as all but 1 is size 1 (e.g. { 10 }, { 1, 10
    # }, { 10, 1, 1 } are all ok). Ignore the batch dimension if there
    # is one.
    output_batch_dim = (model_config.max_batch_size > 0)
    non_one_cnt = 0
    #for dim in output_metadata.shape:
    #    if output_batch_dim:
    #        output_batch_dim = False
    #    elif dim > 1:
    #        non_one_cnt += 1
    #        if non_one_cnt > 1:
    #            raise Exception("expecting model output to be a vector")

    # Model input must have 3 dims, either CHW or HWC (not counting
    # the batch dimension), either CHW or HWC
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

    #if ((input_config.format != mc.ModelInput.FORMAT_NCHW) and
    #    (input_config.format != mc.ModelInput.FORMAT_NHWC)):
    #    raise Exception("unexpected input format " +
    #                    mc.ModelInput.Format.Name(input_config.format) +
    #                    ", expecting " +
    #                    mc.ModelInput.Format.Name(mc.ModelInput.FORMAT_NCHW) +
    #                    " or " +
    #                    mc.ModelInput.Format.Name(mc.ModelInput.FORMAT_NHWC))
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

    outputs = [
        client.InferRequestedOutput(output_name)
    ]

    yield inputs, outputs, model_name, model_version


def convert_http_metadata_config(_metadata, _config):
    _model_metadata = AttrDict(_metadata)
    _model_config = AttrDict(_config)

    return _model_metadata, _model_config

def postprocess(results, output_name, batch_size, batching):
    """
    Post-process results to show classifications.
    """

    output_array = results.as_numpy(output_name)
    print("Output Array: ", output_array)
    print("Output Array Shape: ", output_array.shape)
    if len(output_array) != batch_size:
        raise Exception("expected {} results, got {}".format(
            batch_size, len(output_array)))

    # Include special handling for non-batching models
    for results in output_array:
        print("Results: ", results)
        if not batching:
            results = [results]
        for result in results:
            print("Single Result: :", result)
            if output_array.dtype.type == np.object_:
                cls = "".join(chr(x) for x in result).split(':')
            else:
                cls = result.split(':')
            print("CLS: ", cls)
            print("    {} ({}) = {}".format(cls[0], cls[1], cls[2]))

def trt_infer(blob, model_name, model_version=''):

    # Recognition
    #result = session.run([output_name], {input_name: blob})
    triton_client = httpclient.InferenceServerClient(
                    'localhost:8000', verbose=False, concurrency=1)
    batch_size = len(blob)
    print("Batch Size: ", batch_size)
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

    print("Output Names: ", output_name)
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

    for out_layer in output_name:
        print("Current Output Layer: ", out_layer)
        for inputs, outputs, model_name, model_version in requestGenerator(
            batched_image_data, input_name, out_layer, dtype, model_name, model_version):
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
    #lol = np.

        for response in responses:
            this_id = response.get_response()["id"]
            print("Request {}, batch size {}".format(this_id, batch_size))
            output_array = response.as_numpy(out_layer)
            #print("Output Array: ", output_array)
            print("Output Array Shape: ", output_array.shape)
            #postprocess(response, output_name, batch_size, max_batch_size > 0)

    print("Inference Successful")
    #print("Responses: ", responses)
    #print("Async Requests: ", async_requests)
    output_array = output_array.tolist()
    #print("Output List Length: ", len(output_array))
    return output_array
