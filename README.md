# Demostrate PyTorch with mnist dataset on CPU and GPU
This demo used both high-level API (i.e., sequential model, with GPU) and medium-level API (i.e, subclassing model, with CPU) to build the NN model.

Using tensorflow 2.2 (see another project mnist_digits_CNN_demo in my repositories):

2 sec per training epoch using GPU.


Using pytorch 1.5:

12 sec per training epoch using GPU

33 sec per training epoch using CPU.

<img src="nvidia-smi.png" width="600px" height="150px" />
 
 
<img src="digit demo.png" width="300px" height="300px" />   <img src="one digit predic demo.png" width="300px" height="300px" />
