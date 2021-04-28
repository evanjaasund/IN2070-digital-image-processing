from imageio import imread
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.signal import convolve2d
from scipy.fft import fft2
import time


picture = imread('cow.png', as_gray=True)
N, M = picture.shape

#Function that runs convolve2d on different filter, decided by the number variable.
def plot_oppgave1_3_conv2():
    #Timelist to have control over time elapsed.
    time_list = [0]
    for i in range(1, number*2, 2):
        start_time = time.time()

        filter = np.ones((i, i)) / (i*i)
        picture_domain = convolve2d(picture, filter)

        current_time = time.time()
        elapsed_time = current_time - start_time
        time_list.append(elapsed_time)
    return time_list

#Function that runs fft2/ifft2 on different filter, decided by the number variable.
def plot_oppgave1_3_fft2():
    #Timelist to have control over time elapsed.
    time_list = [0]
    for i in range(1, number*2, 2):
        start_time = time.time()

        filter = np.ones((i, i)) / (i*i)

        #Transform picture and filter over to frequency-domain using fft2-function. Multiplying them and returning them to 
        #picture-domain using ifft2-function.
        frequency_domain = np.fft.fft2(picture)
        frequency_domain_filter = np.fft.fft2(filter, (N, M))
        frequency_domain_convolution = frequency_domain * frequency_domain_filter

        picture_convolved = np.fft.ifft2(frequency_domain_convolution)

        current_time = time.time()
        elapsed_time = current_time - start_time
        time_list.append(elapsed_time)
    return time_list

#Plot-function to arrange and have a good overview over whats going on!
def makePlot():
    conv2_list = plot_oppgave1_3_conv2()
    fft2_list = plot_oppgave1_3_fft2()
    x_pos = np.arange(len(conv2_list))
    plt.title('Comparison: \n Conv2-function & Fft2-function')

    plt.plot(x_pos, conv2_list, color = "red")
    plt.plot(x_pos, fft2_list, color="blue")

    plt.grid()

    plt.legend(["conv2", "fft2"])
    plt.ylabel('Run time')
    plt.xlabel('Filter size')

    plt.show()

number = 10

makePlot()