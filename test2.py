import cream_beta as cream
import multi_neuron_two_depth_nn as mntdnn
import SNN as snn

import tool.Csys as Csys
import tool.progress_bar as pb

import numpy, random, time

Csys.clear()

network = mntdnn.network([2,5,3,1])
# Csys.out(network.check(), Csys.bcolors.OKBLUE)
dataset = cream.dataset.XOR

error = 1
laste = error
i = 0
while abs(error) > 0.1**8:
    network.lrate = min(3, network.lrate+0.000025)
    i+=1
    error = 0
    for data in dataset:
        network.forward(data[0])
        error += sum(cream.Error(network.activ[-1],data[1]))
        network.backpropgation(data[1])

    Csys.clear()
    Csys.out(f"epoch: {i} | {error}", Csys.bcolors.WARNING)

    if numpy.isnan(error):
        Csys.out(network.check(), Csys.bcolors.WARNING)
        break

    laste = error
Csys.out(network.biases, Csys.bcolors.OKCYAN)

while True:
    try:
        input_ = input(" >>> ")
        if (input_) == "weight":
            print(network.weights)

        elif (input_ == "bias"):
            print(network.biases)
        
        else:
            input_ = input_.split(" ")
            if (len(input_) == network.NetworkShape[0]):
                input_ = [eval(i) for i in input_]
                go = True
                for inp in input_:
                    if type(inp) not in [int, float]:
                        go = False
                if (go):
                    network.forward(input_)
                    print(network.activ[-1])

    except Exception as e:
        Csys.out(e, Csys.bcolors.FAIL)