from imageio import imread
import matplotlib.pyplot as plt
import numpy as np
import math


def convolve(filter, picture):
    N, M = picture.shape
    G, H = filter.shape

    kernel = rotate180(filter)

    g_out = np.zeros_like(picture)
    padding = np.zeros((N+G-1, M+H-1))
    padding[(G//2): -(G//2), (H//2): -(H//2)] = picture

    for expand_size in range((G//2) - 1, -1, -1):
        for index in range(expand_size + 1, padding.shape[0] - expand_size - 1):
            padding[index][expand_size] = padding[index][expand_size + 1]
            padding[index][padding.shape[1] - expand_size - 1] = padding[index][padding.shape[1] - expand_size - 2]

        for x in range(expand_size + 1, padding.shape[1] - expand_size - 1):
            padding[expand_size][x] = padding[expand_size + 1][x]
            padding[padding.shape[0] - expand_size - 1][x] = padding[padding.shape[0] - expand_size - 2][x]

        padding[expand_size][expand_size] = picture[0][0]
        padding[expand_size][padding.shape[1] - expand_size - 1] = picture[0][-1]

        padding[padding.shape[0] - expand_size - 1][expand_size] = picture[-1][0]
        padding[padding.shape[0] - expand_size - 1][padding.shape[1] - expand_size - 1] = picture[-1][-1]
            
    for x in range(N):
        for y in range(M):
            g_out[x][y] = ((filter*padding[x: x+len(filter[0]), y: y+len(filter[0])])).sum() #broadcaster filtered med paddingen
    return g_out


def make_blurFilter(dimension):
    sum = dimension**2
    newFilter = np.zeros((dimension, dimension))

    for x in range(newFilter.shape[0]):
        for y in range(newFilter.shape[0]):
            newFilter[x][y] = 1/sum
    return newFilter


def rotate180(matrix):
    n, m = len(matrix), len(matrix[0])
    flipped = np.zeros((n,m))

    for i in range(n):
        for j in range(m):
            flipped[-i-1][-j-1] = matrix[i][j]
    return flipped


def makeGauss(filtersize, sigma):
    filter = np.zeros((filtersize, filtersize))
    A = 0

    for i in range(filter.shape[0]):
        for j in range(filter.shape[0]):
            temp = np.exp(-((i**2+j**2)/(2*sigma**2)))
            filter[i][j] = temp
            A += temp

    for i in range(filter.shape[0]):
        for j in range(filter.shape[0]):
            filter[i][j] = filter[i][j]/A

    return filter

# def makeGauss(size=9, sigma=1):
#     size = int(size) // 2
#     x, y = np.mgrid[-size:size+1, -size:size+1]
#     normal = 1 / (2.0 * np.pi * sigma**2)
#     g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
#     return g


def gradientCalculation(blurredPhoto):
    N, M = blurredPhoto.shape

    magnitudeGradient = np.zeros((N, M))
    directionGradient = np.zeros((N, M))
    
    for i in range(1, N-1):
        for j in range(1, M-1):
            g_x = blurredPhoto[i+1][j] - blurredPhoto[i-1][j]
            g_y = blurredPhoto[i][j+1] - blurredPhoto[i][j-1]
            magnitudeGradient[i][j] = math.sqrt(g_x**2+g_y**2)
            directionGradient[i][j] = math.atan2(g_y, g_x)
    return magnitudeGradient, directionGradient


def thinEdges(magnitudeGradient, directionGradient):
    N, M = magnitudeGradient.shape
    thinned = np.zeros((N, M))
    angles = directionGradient * 180./np.pi
    angles[angles < 0] += 180

    for i in range(1, N-1):
        for j in range(1, M-1):
            a = 255
            b = 255

            if (0 <= angles[i, j] < 22.5) or (157.5 <= angles[i, j] <= 180): #horisontal
                a = magnitudeGradient[i, j+1]
                b = magnitudeGradient[i, j-1]
            
            elif (22.5 <= angles[i, j] < 67.5): #positive diagonal
                a = magnitudeGradient[i+1, j+1]
                b = magnitudeGradient[i-1, j-1]

            elif (67.5 <= angles[i, j] < 112.5): #vertical 
                a = magnitudeGradient[i+1, j]
                b = magnitudeGradient[i-1, j]

            elif (112.5 <= angles[i, j] < 157.5): #negative diagonal
                a = magnitudeGradient[i-1, j+1]
                b = magnitudeGradient[i+1, j-1]
    

            if (magnitudeGradient[i, j] >= a) and (magnitudeGradient[i, j] >= b):
                thinned[i, j] = magnitudeGradient[i, j]
            else:
                thinned[i, j] = 0
    return thinned
 

def threshold(thinned, lowThres, highThres):
    N, M = thinned.shape
    result = np.zeros((N, M))
    # max = 0

    # for i in range(N):
    #     for j in range(M):
    #         if (thinned[i][j] > max):
    #             max = thinned[i][j]

    weak = 130
    strong = 255

    for i in range(N):
        for j in range(M):
            if (thinned[i][j] >= highThres):
                result[i][j] = strong
            elif (lowThres <= thinned[i][j] <= highThres):
                result[i][j] = weak
    
    return result, weak, strong


def hysteresis(picture, weak_pixel, strong_pixel=255):
    N, M = picture.shape

    for x in range(1, N-1):
        for y in range(1, M-1):
            if (picture[x,y] == weak_pixel):
                if ((picture[x+1, y-1] == strong_pixel) or (picture[x+1, y] == strong_pixel) or (picture[x+1, y+1] == strong_pixel) 
                    or (picture[x, y+1] == strong_pixel) or (picture[x, y-1] == strong_pixel)):
                    picture[x,y] = strong_pixel
                else:
                    picture[x,y] = 0
    return picture



picture = imread('cellekjerner.png', as_gray=True)
show = plt.figure()

gaussFilter = makeGauss(11, sigma=3)
processed = convolve(gaussFilter, picture)
m, g = gradientCalculation(processed)
thinnedEdges = thinEdges(m, g)
threshold, weak, strong = threshold(thinnedEdges, 1, 25)
hysteresis = hysteresis(threshold, weak, strong)

blurred3x3 = convolve(gaussFilter, picture)

part1 = show.add_subplot(2, 2, 4)
part2 = show.add_subplot(2, 2, 1)
part3 = show.add_subplot(2, 2, 2)
part4 = show.add_subplot(2, 2, 3)

# part1.imshow(picture, cmap='gray', vmin = 0, vmax = 255)
# part1.title('Original')

part2.imshow(processed, cmap='gray', vmin = 0, vmax = 255)
part2.set_title('Gaussian blur')

part3.imshow(m, cmap='gray', vmin = 0, vmax = 255)
part3.set_title('Gradient magnitude')

part4.imshow(thinnedEdges, cmap='gray', vmin = 0, vmax = 255)
part4.set_title('Thinned gradient magnitude')

part1.imshow(hysteresis, cmap='gray', vmin = 0, vmax = 255)
part1.set_title('Hysteresis w/ sigma = 3, T_l = 1, T_h = 25')

plt.show()
