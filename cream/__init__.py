
# Cream ( CoRy Ego Ai Model )
# Version: 0.0.1
# Developer: @mangto (github)
# Contact: mangto0701@gmail.com

import cream.engine.cream_beta as cream
import cream.engine.one_neuron_snn as onsnn
import cream.engine.SNN as snn
import cream.engine.multi_neuron_two_depth_nn as mntdnn
import cream.engine.convolutional_neural_network as cnn

import cream.Functions as functions
import cream.tool.Csys as csys
import cream.tool.datasets as datasets
import cream.tool.progress_bar as progress_bar
import cream.Functions.cnn.kernel as kernel
from cream.Functions.cnn.convolution import *
import cream.Functions.cnn.pooling as pool
from cream.Functions.cnn.padding import *

import cream.visualizer

print(f"CREAM Beta Version 0.0.1 by @mangto\nIf you find bugs or something to change, please contact 'mangto0701@gmail.com'")