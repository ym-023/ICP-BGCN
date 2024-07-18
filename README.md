# ICP-BGCN
PyTorch implementation of "Integrating Message Content and Propagation Path for Enhanced False Information Detection Using Bidirectional Graph Convolutional Neural Networks"

# Dependencies  
python 3.11  
pytorch 2.1.0  
pytorch_geometric 2.4.0  

# Datasets
Data processing of Twitter15 and Twitter16 social interaction graphs follows [BiGCN](https://github.com/TianBian95/BiGCN).

The raw datasets can be respectively downloaded from https://www.dropbox.com/s/7ewzdrbelpmrnxu/rumdetect2017.zip?dl=0.

# Visualization of information dissemination networks
The information dissemination network visualization graphs for all events in the Twitter15 and Twitter16 datasets are located in the _propagation_graph_ folder.

# Run
With two arguments, first stands for dataset's name, the latter is the number of iterations 
'''
python ICP-BGCN.py Twitter15 100
python ICP-BGCN.py Twitter16 100
'''
