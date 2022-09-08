import numpy
import multiprocessing

mask_type = [numpy.array, numpy.ndarray]

def convolution(image:numpy.array, mask:numpy.array):
    
    mask_shape = mask.shape
    image_shape = image.shape
    result_shape = tuple(numpy.array(image_shape) - numpy.array(mask_shape) + 1)
    result = numpy.zeros(result_shape)
    mask = numpy.array(mask)

    for h in range(0, result_shape[0]):
        for w in range(0, result_shape[1]):
            tmp = image[h:h+mask_shape[0], w:w+mask_shape[1]]
            res = numpy.abs(numpy.sum(mask * tmp))
            result[h][w] = res

    return result

def convolutions(image, *masks):
    result = numpy.array([convolution(image, mask) for mask in masks])
    return result

    

def threshold(convolutions:list, threshold:int=100):
    convolut = numpy.sum(convolutions, axis=0)
    thr_result = numpy.zeros(convolutions[0].shape)
    thr_result[convolut>threshold] = 1
    return thr_result