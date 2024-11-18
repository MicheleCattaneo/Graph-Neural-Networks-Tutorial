# Graph Neural Networks Tutorial

This repo contains a collection of tutorials on some basic concepts regarding neural graph processing. In particular: 

- `gnn.ipynb` aims at presenting some basic concepts about graph neural networks and how PyTorch Geometric (PyG) can be used to define custom GNN layers. Here, the graph attention network (GAT) is written from scratch starting by the message passing framework of PyG and applied on a semi-supervised node classification task.
- `stgnn.ipynb` focuses instead on the use of graph neural networks for time series forecasting. Relying on Torch Spatiotemporal (tsl), we try to forecast the air quality in China recorded by a network of sensors over time. 
- `graph-shift.ipynb` focuses on graph-shift operators and how they can be used to obtain graph convolutional networks. 
