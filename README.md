# Graph Neural Networks Tutorial

This repo contains a collection of tutorials on some basic concepts regarding neural graph processing. In particular: 

- `gnn.ipynb` aims at presenting some basic concepts about graph neural networks and how PyTorch Geometric (<a href="https://pyg.org/" target="_blank">PyG</a>)
 can be used to define custom GNN layers. Here, the graph attention network (GAT) is written from scratch starting from the message passing framework of PyG and applied on a semi-supervised node classification task.
- `stgnn.ipynb` focuses instead on the use of graph neural networks for time series forecasting. Relying on Torch Spatiotemporal (<a href="https://torch-spatiotemporal.readthedocs.io/en/latest/index.html" target="_blank">tsl</a>), we try to forecast the air quality in China recorded by a network of sensors over time. 
- `graph-shift.ipynb` focuses on graph-shift operators and how they can be used to obtain graph convolutional networks.

#### Quickstart: 
- Create a dedicated python envoronment, e.g with `conda create --name gnn_tutorial python=3.10`.
- Deactivate any current environment with `conda deactivate` and activate the newly created one with `conda activate gnn_tutorial`.
- Install the requirements with `bash requirements.sh`. These installations assume a CUDA-enabled environment. To use a CPU, change the suffix `cu121` with `cpu` in the commands.

