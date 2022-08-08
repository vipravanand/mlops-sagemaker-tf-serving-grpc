

import os
import boto3
import json
import numpy as np

# grab environment variables
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
runtime= boto3.client('runtime.sagemaker')

def lambda_handler(event, context):
    print("Received event: " + json.dumps(event, indent=2))
    mae_threshold = 0.275  #Mean absolute error Values higher than this is classified as an anomaly
    
    data = json.loads(json.dumps(event))
    payload = data['data']
    payload = np.array(payload)
    print(payload)
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                       ContentType="application/npy",
                                       Body=payload.tobytes())

    response = json.loads(response['Body'].read().decode())
    print(response)
    prediction = np.array(response['predictions'])
    
    mae = float(np.mean(np.abs( payload - prediction)))
    anomaly = mae > mae_threshold
    response_dict = {'mae ':mae, 'anomaly ': anomaly}
    
    
    return json.dumps(response_dict)
