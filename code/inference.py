print('******* in inference.py *******')
import tensorflow as tf
print(f'TensorFlow version is: {tf.version.VERSION}')

import json
import numpy as np
from collections import namedtuple

import grpc
from tensorflow.compat.v1 import make_tensor_proto
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

import os

    
MAX_GRPC_MESSAGE_LENGTH = 512 * 1024 * 1024

HEIGHT = 224
WIDTH  = 224

# Restrict memory growth on GPU's
physical_gpus = tf.config.experimental.list_physical_devices('GPU')
if physical_gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in physical_gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(physical_gpus), 'Physical GPUs,', len(logical_gpus), 'Logical GPUs')
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
else:
    print('**** NO physical GPUs')


num_inferences = 0
print(f'num_inferences: {num_inferences}')

Context = namedtuple('Context',
                     'model_name, model_version, method, rest_uri, grpc_uri, '
                     'custom_attributes, request_content_type, accept_header')

def handler(data, context):


    if context.request_content_type in ('application/x-npy', "application/npy"):

        data = np.frombuffer(data.read())
        
        data = np.array(data).reshape(-1,1,4)
        
        prediction = _predict_using_grpc(context, data)
    
        response_content_type = context.accept_header
        
        return prediction, response_content_type
    
    return 

def _return_error(code, message):
    raise ValueError('Error: {}, {}'.format(str(code), message))

def _predict_using_grpc(context, instance):

    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'auto_encoder_model'
    request.model_spec.signature_name = 'serving_default'

    request.inputs['input_1'].CopyFrom(make_tensor_proto(instance, dtype = tf.float32))
    options = [
        ('grpc.max_send_message_length', MAX_GRPC_MESSAGE_LENGTH),
        ('grpc.max_receive_message_length', MAX_GRPC_MESSAGE_LENGTH)
    ]
    channel = grpc.insecure_channel(f'0.0.0.0:{context.grpc_port}', options=options)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    result_future = stub.Predict.future(request, 30)  # 5 seconds  

    output_tensor_proto = result_future.result().outputs['time_distributed']
    
    output_shape = [dim.size for dim in output_tensor_proto.tensor_shape.dim]
    
    
    output_np = np.array(output_tensor_proto.float_val).reshape(output_shape)
    prediction_json = {'predictions': output_np.tolist()}
    channel.close()
    return json.dumps(prediction_json)
    
\