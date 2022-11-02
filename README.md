# A Comparative Study of Graph Representations for Few-Shot Link Prediction

link to the [report](https://github.com/abdalgader-a/MetaLearning-for-graph-representation-with-application-to-drug-discovery/blob/master/Note.pdf).

# Enviroment

The requierd packages for the enviroment can be founded here [requirements.txt](Meta-Graph/requirements.txt). The main packages are:
* torch==1.4.0
* torch-scatter==latest
* torch-sparse==latest
* torch-spline-conv==latest
* torch-cluster==latest
* torch-geometric==1.5.0


In CPU case the installation command is,

```
pip install torch-scatter==latest+cpu torch-sparse==latest+cpu torch-cluster==latest+cpu torch_spline_conv==latest+cpu -f https://pytorch-geometric.com/whl/torch-1.4.0.html 
```
or, if you use GPU (put attention to compatibility of CUDA versions)

```
pip install torch-scatter==latest+cu101 torch-sparse==latest+cu101 torch-cluster==latest+cu101 torch_spline_conv==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
```
Then install Pytorch Geometric v1.5.0,
```
pip install torch-geometric==1.5.0
```
