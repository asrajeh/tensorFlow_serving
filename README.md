# TensorFlow Serving
Ready to run example of TensorFlow serving, written by Alrajeh

Reference: https://www.tensorflow.org/tfx/serving/tutorials/Serving_REST_simple

```
./install.sh # run for one time to install the requirements

python model.py # train and save a model

./server.sh # start the server

python request.py # send web request to the model

./stop.sh # stop the server
```
