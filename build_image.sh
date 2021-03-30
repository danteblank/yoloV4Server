#!/bin/bash

nvidia-docker build -t yolov4_inference_api_gpu -f ./docker/dockerfile .