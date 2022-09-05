import cream, time, numpy

cream.csys.clear()

start = time.time()
network = cream.mntdnn.network([784,16,10], ActivationFunction=cream.functions.msigmoid,LearningRate=1.0)
dataset = cream.datasets.mnist_train
test_dataset = cream.datasets.mnist_test

accuracy = 1
epoch = 0
count = len((dataset[1][0]))
test_count = len(dataset[1][0])

while accuracy < 99 or epoch==0:
    epoch += 1
    pb = cream.progress_bar.bar(60,title=f"epoch {epoch}", design="=")
    pb.start()

    for i in range(count):
        pb.update((i+1)/count*100)
        image, label = dataset[0][0][i], dataset[1][0][i]

        network.forward(numpy.array(image)/255)
        network.backpropgation(numpy.eye(10)[label])

    pb.end()

    pb = cream.progress_bar.bar(60,title=f"test  {epoch}", design="=")
    pb.start()
    
    accuracy = 0

    for i in range(test_count):
        pb.update((i+1)/test_count*100)
        image, label = dataset[0][0][i], dataset[1][0][i]

        network.forward(numpy.array(image)/255)
        accuracy += numpy.argmax(network.activ[-1]) == numpy.argmax(label)
    
    accuracy = accuracy / test_count * 100
    pb.end()
    cream.csys.out(f"accuracy: {accuracy}%", cream.csys.bcolors.WARNING)