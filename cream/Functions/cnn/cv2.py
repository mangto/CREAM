import cv2, numpy, math

def automatic_contrast(image):
    result = cv2.equalizeHist(image)
    return result

def automatic_brightness(image:numpy.array, target_average=170, min_average=100):
    type_ = type(image)
    shape = image.shape
    dimension = len(shape)

    assert type_ == numpy.array or type_ == numpy.ndarray, f"type of image have to be numpy.array or numpy.ndarray, taken: {type_}"
    # assert dimension == 2, f"image's dimension have to be 2d, taken: {dimension}"

    count = math.prod(shape)
    sum_ = numpy.sum(image)
    average = max(sum_/count, min_average)
    multiplier = target_average / average

    image = image * multiplier

    return image

def contrast(image, alpha=1.0):
    result = numpy.clip((1+alpha)*image - 128*alpha, 0, 255).astype(numpy.uint8)
    return result

def automatic_brightness_and_contrast(image, clip_hist_percent=25):

    # Calculate grayscale histogram
    hist = cv2.calcHist([image],[0],None,[256],[0,256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return auto_result
