import sys, os, numpy, multiprocessing, pickle, time


def convolution_h(argv:list):
    image = argv[0]
    mask = argv[1]
    h = argv[2]
    result_shape = argv[3]
    mask_shape = argv[4]
    result = numpy.zeros((result_shape[1], ))

    for w in range(0, result_shape[1]):
        tmp = image[h:h+mask_shape[0], w:w+mask_shape[1]]
        res = numpy.abs(numpy.sum(mask * tmp))
        result[w] = res

    return result

if __name__ == "__main__":
    argv = sys.argv
    img_loc = argv[1]
    mask_loc =  argv[2]
    return_loc = argv[3]


    if (os.path.isfile(img_loc) and os.path.isfile(mask_loc)):
        array = numpy.array
        image = pickle.load(open(img_loc,"rb"))
        mask = pickle.load(open(mask_loc,"rb"))
        mask_shape = mask.shape
        image_shape = image.shape
        result_shape = tuple(numpy.array(image_shape) - numpy.array(mask_shape) + 1)
        task = [[image,mask,h,result_shape,mask_shape] for h in range(result_shape[0])] 

        start = time.time()

        cpu_count = int(multiprocessing.cpu_count()//1.5)
        pool = multiprocessing.Pool(6)

        returner = pool.map_async(convolution_h, task)
        result = numpy.array(returner)

        pickle.dump(result, open(return_loc,"wb"))