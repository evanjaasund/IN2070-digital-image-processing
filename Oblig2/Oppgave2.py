from imageio import imread
from numpy.core.fromnumeric import compress
from scipy.fft import fft2
from scipy.signal import convolve2d
from numpy.lib.histograms import histogram

import numpy as np
import matplotlib.pyplot as plt

#Function for arranging the whole compression and decompression for the input picture (orgPicture). Function also takes integerparameter q, 0.1 if not assinged.
def compression(orgPicture, q=0.1):
    N, M = orgPicture.shape
    #Q-array from the mandatory assignment.
    Q = np.array([[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55], [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62], [
                 18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92], [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]])
    #Multiplying q and Q for later use. This is done as it has been instructed from the mandatory assingment.
    qQ = q*Q

    #Extracting 128 from each pixel to reduce the average pixel from expected 128 to 0.
    picture = orgPicture - 128
    
    #Transforming the picture using the transformDCT-function. As well as sending the qQ product. After the function has runned, 
    #a reverse transformation reverseDCT is called.
    transformedPicture = transformDCT(picture, qQ)
    transformedBack = reverseDCT(transformedPicture, qQ)

    #Adding back the 128 subtracted earlier.
    transformedBack = transformedBack + 128

    return transformedPicture, transformedBack

#Function that does a discrete cosine transform (DCT). Found in lecture slides.
def transformDCT(picture, qQ):
    N, M = picture.shape
    transformedPicture = np.zeros((N, M))

    #Divide the picture into 8x8 blocks.
    for i in range(0, N, 8):
        for j in range(0, M, 8):
            #Iterating through the elements in 8x8 blocks.
            for u in range(8):
                for v in range(8):
                    #Resetting the sum for next element.
                    sum = 0
                    #Iterating once more for calculating the sum for each element by adding the sum together for all remaining elements.
                    for x in range(8):
                        for y in range(8):
                            #Using a formula for sum found in lecture slides.
                            sum += picture[x+i, y+j] * np.cos(
                                ((2*x+1)*(u)*np.pi)/16) * np.cos(((2*y+1)*(v)*np.pi)/16)
                    #Assingning new values with use of a formula found in lecture slides. This divides all elements by corresponding element in qQ-matrix.
                    transformedPicture[u+i, v+j] = np.round(
                        (1/4 * c(u) * c(v) * sum)/(qQ[u, v]))

    return transformedPicture

#Function that does a inverse discrete cosine transform (IDCT). Found in lecture slides.
def reverseDCT(transformedPicture, qQ):
    N, M = transformedPicture.shape

    #Creating a immediate zero-array as i dont want to change input-array as well as a output-array. Both in size corresponding to input-array. 
    immediate = np.zeros((N, M))
    reversedBack = np.zeros((N, M))

    #Divide the picture into 8x8 blocks.
    for i in range(0, N, 8):
        for j in range(0, M, 8):
            #Iterating through the elements in 8x8 blocks.
            for x in range(8):
                for y in range(8):
                    #Resetting the sum for next element.
                    sum = 0
                    #Iterating once more for calculating the sum for each element by adding the sum together for all remaining elements.
                    for u in range(8):
                        for v in range(8):
                            #Using a formula for sum found in lecture slides for reverse DCT. As well as assigning to immediate-array instead of overwriting input-array.
                            #Here the elements are also multiplied back up with the corresponding elements in the qQ-matrix.
                            immediate[u+i, v +
                                      j] = transformedPicture[u+i, v+j] * qQ[u, v]
                            sum += c(u) * c(v) * immediate[u+i, v+j] * np.cos(
                                ((2*x+1)*(u)*np.pi)/16) * np.cos(((2*y+1)*(v)*np.pi)/16)
                    #Assingning new values with use of a formula found in lecture slides.
                    reversedBack[x+i, y+j] = np.round(1/4 * sum, 0)

    return reversedBack

#Function for calculating the entropy for a picture. Found in lecture slides.
def calculateEntropy(normalizedHistogram):
    sum = 0
    for i in range(len(normalizedHistogram)):
        if (normalizedHistogram[i] != 0):
            sum -= normalizedHistogram[i] * np.log2(1/normalizedHistogram[i])
    return sum

#Function for calculating the percentage removed by compression. Uses a formula for relative redundance times a 100 for percentage. Found in lecture slides.
def calculatePercentageRemoved(decompressedEntropy, compressedEntropy):
    datasize = 0
    R = 1 - (compressedEntropy/decompressedEntropy)
    datasize = 100*R
    return datasize

#Function for returning either 1/sqrt(2) or 1 depending on input. Found in lecture slides.
def c(number):
    if number == 0:
        return 1/np.sqrt(2)
    else:
        return 1

#Function that finds entropy using calculateEntropy-function found in the lecture slides.
def findEntropy(picture):
    histogram = makeHistogram(picture)
    normHistogram = normHisto(histogram)
    entropy = calculateEntropy(normHistogram)
    return entropy

#Function that makes a histogram, used in the first mandatory assignments.
def makeHistogram(f):
    N, M = f.shape
    G = 500
    hist = [0] * G
    values = []

    for i in range(N):
        for j in range(M):
            value = int(f[i, j]) + 128
            if(value in range(0, G-1)):
                hist[value] = hist[value] + 1
                if (value not in values):
                    values.append(value)
    return hist

#Function that makes a normalized histogram, used in the first mandatory assignments.
def normHisto(hist):
    tot = sum(hist)
    p = hist.copy()
    for i in range(len(p)):
        p[i] = hist[i]/tot
    return p

#Function to organize plotting. This is to make the program easier to use and understand. Uses picture path to assigning picture.
def plot(picturePath):
    orgPicture = imread(picturePath, as_gray=True)
    orgEntropy = findEntropy(orgPicture)
    q_list = [0.1, 0.5, 2, 8, 32]

    #Arrays for saving values when iterating through q_list.
    tBacklist = []
    cRatelist = []
    pRemovedlist = []

    #Iterating through the q´s to assign values to plot.
    for q in q_list:
        transformedPicture, transformedBack = compression(orgPicture, q)
        tBacklist.append(transformedBack)

        #Finding the entropy for both original and the transformed picture using formulas found in the lectureslides.
        newEntropy = findEntropy(transformedPicture)

        #Calculating the compressionrate using a formula found in the lectureslides.
        compressionRate = orgEntropy/newEntropy
        cRatelist.append(compressionRate)

        #Calculating the persantage removed by compression using a formula found in the lectureslides.
        percentageRemoved = calculatePercentageRemoved(orgEntropy, newEntropy)
        pRemovedlist.append(percentageRemoved)

    f, axarr = plt.subplots(2, 3)

    print("Percentage removed by compression: {:.3f} %".format(
        pRemovedlist[0]))
    print("Compression rate: {:.3f} ".format(cRatelist[0]))

    print("Percentage removed by compression: {:.3f} %".format(
        pRemovedlist[1]))
    print("Compression rate: {:.3f} ".format(cRatelist[1]))

    print("Percentage removed by compression: {:.3f} %".format(
        pRemovedlist[2]))
    print("Compression rate: {:.3f} ".format(cRatelist[2]))

    print("Percentage removed by compression: {:.3f} %".format(
        pRemovedlist[3]))
    print("Compression rate: {:.3f} ".format(cRatelist[3]))

    print("Percentage removed by compression: {:.3f} %".format(
        pRemovedlist[4]))
    print("Compression rate: {:.3f} ".format(cRatelist[4]))

    axarr[0, 0].imshow(orgPicture, cmap='gray')
    axarr[0, 0].set_title("Original")

    axarr[0, 1].imshow(tBacklist[0], cmap='gray')
    axarr[0, 1].set_title("q = 0.1")

    axarr[0, 2].imshow(tBacklist[1], cmap='gray')
    axarr[0, 2].set_title("q = 0.5")

    axarr[1, 0].imshow(tBacklist[2], cmap='gray')
    axarr[1, 0].set_title("q = 2")

    axarr[1, 1].imshow(tBacklist[3], cmap='gray')
    axarr[1, 1].set_title("q = 8")

    axarr[1, 2].imshow(tBacklist[4], cmap='gray')
    axarr[1, 2].set_title("q = 32")

    plt.show()

#Assign path to desired picture.
plot("/Users/evan/Desktop/4. Semester/IN2070/Oblig2/uio.png")
