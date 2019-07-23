#!/bin/bash

# Add TensorFlow Serving distribution URI as a package source:
echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | sudo tee /etc/apt/sources.list.d/tensorflow-serving.list && \
curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | sudo apt-key add -
sudo apt-get update

# Install TensorFlow Serving
sudo apt-get install tensorflow-model-server

pip install -q requests
