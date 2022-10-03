import numpy
import numba

from numba.core.errors import NumbaWarning, NumbaDeprecationWarning
import warnings
warnings.simplefilter('ignore', category=NumbaWarning)
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)

from cream.Functions.cnn.padding import *

mask_type = [numpy.array, numpy.ndarray]

class convolution_layer:
    def __init__(self, kernels:list, pool=None, padding:int=None, image_count:int=1):\

        assert (padding_type := type(padding)) == int, f"Type of padding have to be 'int', taken type: {padding_type}"
        assert padding >= 0, f"'padding' have to be positive number"
        assert (image_count_type := type(image_count)) == int, f"Type of image_count have to be 'int', taken type: {image_count_type}"


        self.kernels = kernels
        self.pool = pool
        self.padding = padding
        self.image_count = image_count

    def __str__(self):
        result = f'''
        | type: convolution layer
        | kernels: {[array.tolist() for array in self.kernels]}
        | pool: {self.pool}
        | padding: {self.padding}
        | image count: {self.image_count}
        '''
        return result


    @numba.jit
    def forward(self, images:list):

        assert (image_count := len(images)) == self.image_count, f"Count of image have to be {self.image_count}, taken: {image_count}"
        

        result = []

        for i in range(self.image_count):
            image = images[i]

            # add padding
            if self.padding: image = pad(image, self.padding)

            res = convolutions(image, tuple(self.kernels)) # result
            result.append(res)

        # 
        # pooling
        # 

        return result

@numba.jit
def convolution(image:numpy.array, mask:numpy.array):
    
    mask_shape = mask.shape
    image_shape = image.shape
    result_shape = tuple(numpy.array(image_shape) - numpy.array(mask_shape) + 1)
    result = numpy.zeros(result_shape)
    mask = numpy.array(mask)

    for h in range(0, result_shape[0]):
        for w in range(0, result_shape[1]):
            tmp = image[h:h+mask_shape[0], w:w+mask_shape[1]] # load part of image to calculate
            res = numpy.abs(numpy.sum(mask * tmp))
            result[h][w] = res

    return result

@numba.jit
def multiple_image_convolution(images:list, masks):
    result = [convolutions(image, masks) for image in images]
    return result

@numba.jit
def convolutions(image, *masks):

    if (type(masks[0]) == tuple and len(masks) == 1): masks = masks[0]

    result = [convolution(image, mask) for mask in masks]
    result = numpy.array(result)
    return result



@numba.jit
def add_convolutions(convolutions:list):
    return numpy.sum(convolutions, axis=0)



@numba.jit
def threshold(convolutions:list, threshold:int=100):
    convolut = numpy.sum(convolutions, axis=0)
    thr_result = numpy.zeros(convolutions[0].shape)
    thr_result[convolut>threshold] = 1
    return thr_result

@numba.jit
def multiple_convolutions_threshold(convolutions:list, threshold:int=100):
    result = []
    for convolution in convolutions:
        convolut = numpy.sum(convolution, axis=0)
        thr_result = numpy.zeros(convolution[0].shape)
        thr_result[convolut>threshold] = 1
        result.append(thr_result)
    return result