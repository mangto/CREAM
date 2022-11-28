import cream, numpy

network = cream.network()
network.add(cream.layer.Dense(20, cream.functions.sigmoid, InputShape=1000))
# network.add(cream.layer.Dropout(0.2))
network.add(cream.layer.Dense(10, cream.functions.sigmoid))
network.compile()

print(network.foreward(numpy.random.randn(1000)))