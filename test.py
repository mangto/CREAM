import cream, random, numpy

dataset = cream.datasets.XOR
network = cream.network()
network.add(cream.layer.Dense(5, cream.functions.ReLU, InputShape=2))
network.add(cream.layer.Dense(1, cream.functions.ReLU))
network.compile()

print(dataset)
error = 1
epoch = 0
while (error > 0.1**15 and epoch <= 10000):
    error = 0
    for data in dataset:
        network.forward(data[0])
        network.backward(data[1])
        error += sum(cream.functions.Error(network.activ[-1], data[1]))

    cream.csys.out(f"epoch: {epoch:>6} | error: {error}", cream.csys.OKCYAN)
    epoch += 1
    
    if (numpy.isnan(error)):
        cream.csys.stop()

for layer in network.layers:
    cream.csys.out(layer.weights, cream.csys.OKBLUE)
    cream.csys.out(layer.biases, cream.csys.OKGREEN)
cream.csys.stop("yo..")