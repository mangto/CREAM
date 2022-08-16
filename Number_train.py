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

network = cream.snn([784, 20, 10], cream.sigmoid, LearningRate=0.01)
images, labels = get_mnist()
count = len(images)
accuracy = 0
epoch = 0


while accuracy < 99.9:
    correct = 0
    i = 0
    epoch += 1

    bar = pb.bar(width = 40, design=pb.box, title=f"epoch {epoch}")
    bar.start()

    for image, label in zip(images,labels):
        i += 1
        bar.update(i/count*100)
        image = numpy.reshape(image, (1, 784))[0]
        network.forward(image)

        if (numpy.argmax(network.activations[-1]) == numpy.argmax(label)):
            correct += 1

        network.backpropgation(label)

    accuracy = round(correct/count*100, 2)

    print(f"accuracy: {accuracy}%")
pickle.dump(network.weights, open(".\\Number_data\\weights.dat","wb"))
pickle.dump(network.biases, open(".\\Number_data\\biases.dat","wb"))

Csys.stop()