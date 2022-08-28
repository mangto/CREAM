import numpy, pickle
import SNN as cream
import tool.Csys as Csys
import tool.progress_bar as pb

Csys.clear()

def get_mnist():
    with numpy.load(f"mnist.npz") as f:
        images, labels = f["x_train"], f["y_train"]
    images = images.astype("float32") / 255
    images = numpy.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))
    labels = numpy.eye(10)[labels]
    return images, labels

network = cream.snn([784, 20, 10], cream.sigmoid, LearningRate=0.3)
network.load_weight(pickle.load(open(".\\Number_data\\weights.dat","rb")))
network.load_bias(pickle.load(open(".\\Number_data\\biases.dat","rb")))
images, labels = get_mnist()
count = len(images)
accuracy = 0
epoch = 0


while accuracy > 100 or epoch == 0:
    correct = 0
    i = 0
    epoch += 1

    bar = pb.bar(width = 70, design=pb.box, title=f"epoch {epoch}")
    bar.start()

    for image, label in zip(images,labels):
        i += 1
        bar.update(i/count*100)
        image = numpy.reshape(image, (1, 784))[0]
        network.forward(image)

        correct += sum(cream.Error(network.activations[-1],label))

        network.backpropgation(label)

    accuracy = correct

    print(f"Total Error: {accuracy}")
    pickle.dump(network.weights, open(".\\Number_data\\weights.dat","wb"))
    pickle.dump(network.biases, open(".\\Number_data\\biases.dat","wb"))

Csys.stop()