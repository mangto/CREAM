import cream_beta as cream
import Csys


Csys.clear()

net = cream.Network([2,5,1],function=cream.ReLU,learning_rate=0.3)

'''open(f'{os.getcwd()}\\visualizer\\data\\weights.dat','w',encoding='utf-8').write(str(net.weights))
cream.visualizer.run(f'{os.getcwd()}\\visualizer\\data\\weights.dat')'''
net.train(1, cream.dataset.XOR,100000)