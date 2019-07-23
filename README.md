# TensorFlow Serving
Ready ro run example of tensorFlow serving, written by Alrajeh

Rrefrence: https://www.tensorflow.org/tfx/serving/tutorials/Serving_REST_simple

```
./install.sh # run for one time to install the requirments

python model.py # train and save a model

./server.sh # start the server

python request.py # send web request to the model

kill `cat server.pid` # kill the server
```
