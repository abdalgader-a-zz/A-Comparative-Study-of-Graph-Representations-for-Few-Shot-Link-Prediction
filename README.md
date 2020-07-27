# Meta-Learning for Graph Representation with Application to Drug Discovery

 "write the description here "

# Enviroment

The requierd packages for the enviroment can be founded here [requirements.txt](Meta-Graph/requirements.txt), then you should use the following command, if you use CPU.

```
pip install torch-scatter==latest+cpu torch-sparse==latest+cpu torch-cluster==latest+cpu torch_spline_conv==latest+cpu -f https://pytorch-geometric.com/whl/torch-1.4.0.html 
```
or, if you use GPU (replace cu101 with the your cuda version)

```
pip install torch-scatter==latest+cu101 torch-sparse==latest+cu101 torch-cluster==latest+cu101 torch_spline_conv==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
```
Then install Pytorch Geometric v1.5.0 using the following command
```
pip install torch-geometric==1.5.0
```
