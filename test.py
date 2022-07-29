import cream_beta as cream
import Csys

Csys.clear()

net = cream.Network([2,2,2,2,2],function=cream.sigmoid)
print(net.weights)

net.train(5, cream.dataset.Reverse,1)

Csys.stop()