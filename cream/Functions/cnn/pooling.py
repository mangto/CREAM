import numpy



def average(image:numpy.array, pool_size:tuple=(2, 2)):

    image_size = image.shape
    #assert (image_size[0] % pool_size[0] == 0 and image_size[1] % pool_size[1] == 0), "wrong pool size"

    x, y = image_size
    new_x, new_y = x//2, y//2
    
    result = numpy.mean(image.reshape(new_x, pool_size[0], new_y, pool_size[1]), axis=(1, 3))
    return result


def max(image:numpy.array, pool_size:tuple=(2,2)):

    image_size = image.shape
    #assert (image_size[0] % pool_size[0] == 0 and image_size[1] % pool_size[1] == 0), "wrong pool size"

    x, y = image_size
    new_x, new_y = x//2, y//2

    result = image[:new_x*pool_size[0], :new_y*pool_size[1]].reshape(new_x, pool_size[0], new_y, pool_size[1]).max(axis=(1,3))
    return result