# Pytorch-based-Implementation-for-Label-aware-Graph-Convolution-Networks
Source code for "PyTorch-based implementation of label-aware graph representation for multi-class trajectory prediction".
<p align="center">
  <img src="res/graph.png"/> 
</p>

## Training and validation
The evaluation is calculated based on bi-variant distributions for the 2D traffic trajectory and l2 norm for 3D skeleton trajectory prediction.

To train on traffic trajectory, run: python train_2D3D.py --dataset 2D <br>
To train on skeleton trajectory, run: python train_2D3D.py --dataset 3D 
