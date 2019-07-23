#!/bin/bash

MODEL_DIR=`python -c 'import tempfile; print(tempfile.gettempdir())'`

nohup tensorflow_model_server \
  --rest_api_port=8501 \
  --model_name=fashion_model \
  --model_base_path="${MODEL_DIR}" >server.log 2>&1 &

echo $! > server.pid
