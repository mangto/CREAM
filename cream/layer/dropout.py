import numpy, os

class Dropout:
    def __init__(self, percentage:float):
        self.percentage = percentage
        self.InputShape = None

        self.size:int
        self.weights:numpy.ndarray
        self.biases:numpy.ndarray

        # self.backpropagation = False

        return

    def __call__(self, previous_activation:list | numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
        return (self.do(previous_activation))

    def get_weights(self):
        return self.weights

    def generate(self, previous_shape:int | tuple) -> None:
        
        self.size = previous_shape
        self.weights = numpy.ones((self.size))
        self.biases = numpy.zeros((self.size))

        return

    def do(self, previous_activation:list | numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
        self.weights = numpy.ones((self.size, ))
        random = numpy.abs(numpy.random.rand(self.size, ))
        self.weights[random<=self.percentage] = 0
        result = previous_activation * self.weights

        return previous_activation, result

    def backpropagation(self, args:dict) -> list | numpy.ndarray:
        delta = args.get('delta')
        # delta *= self.weights

        print(delta)
        print(self.weights)

        os.system("pause")

        return delta