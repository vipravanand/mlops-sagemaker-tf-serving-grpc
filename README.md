## Project Overview

-Deploy a pretrained tensorflow SavedModel as Sagemaker endpoint usng the tf-serving api 
-Create a AWS lambda function to expose a REST API that invokes the above created sagemaker endpoint

Note  ; Pretrained model is a AutoEncoder Tensorflow model configured to act as a Anomaly Detector based on MAE ( Mean Absolute Error) based threshold)

### Sagemaker Endpoint 
The Autoencoder SavedModel is compressed as a tar file for upload omn s3 bucket. 
The Model is deployed as sagemaker endpoint. 
The inference.py file preproceses the incoming request for inferecing usung the grpc protocol , which is considered more efficient than REST Api for large data volumne inferencing 


### AWS Lambda Function - API Gateway

The AWS lambda function is deployed using API Gateway to invoke the sagemaker endpoint deployed above

### File System 

sagemaker-tf-serving-grpc-deployment.ipynb  - > Notebook for Model Deployment

code/inference,py  -> Entrypoint for data preprocessing and inferecing using grpc protocol

lambda_function.py -> for creating AWS lambda function that is configured thrpugh API Gateway as Rest Api to invoke sagemaker endponts


