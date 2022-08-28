import cream_beta as cream
import tool.Csys as Csys
import tool.progress_bar as pb
import numpy, random, time
from threading import Thread

Csys.clear()

network = cream.network([1,2,2,1], ActivationFunction=cream.Leaky_ReLU,LearningRate=0.3)
