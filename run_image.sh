#!/bin/bash

docker run --name=YOLO4API --runtime=nvidia -p="9999:9999" yolov4_inference_api_gpu:latest