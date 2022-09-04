import cream

network = cream.mntdnn.network([2,5,1])
dataset = cream.datasets.XOR

error = 1
laste = error
i = 0
while abs(error) > 0.1**8:
    network.lrate = min(3, network.lrate+0.000025)
    i+=1
    error = 0
    for data in dataset:
        network.forward(data[0])
        error += sum(cream.functions.Error(network.activ[-1],data[1]))
        network.backpropgation(data[1])

    cream.Csys.out(f"epoch: {i} | {error}", cream.csys.bcolors.WARNING)

    laste = error

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
        cream.csys.out(e, cream.csys.bcolors.FAIL)