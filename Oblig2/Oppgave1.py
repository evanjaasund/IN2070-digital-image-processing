from imageio import imread
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.signal import convolve2d
from scipy.fft import fft2
print("OK")

picture = imread('cow.png', as_gray=True)
N, M = picture.shape


def make_Filter(dimension):
    newFilter = np.ones((dimension, dimension)) / (dimension*dimension)
    return newFilter


def plot_oppgave1_1():
    filter = make_Filter(15)
    V, W = filter.shape

    # 'same'-parameteret fjerner den sorte kanten rundt bilde.
    convolved = convolve2d(picture, filter, 'same')

    # Sender bilde og filteret inn i frekvensdomenet, hvor de ganges sammen.
    frequency_domain = np.fft.fft2(picture)

    # Utvider filteret i samme fft2-kallet, istedet for å utvide først (linje 27 og 28).
    frequency_domain_filter = np.fft.fft2(filter, (N, M))
    frequency_domain_convolution = frequency_domain * frequency_domain_filter

    # Returnerer produktet til bildedomenet igjen.
    picture_convolved = np.fft.ifft2(frequency_domain_convolution)

    show = plt.figure()
    part1 = show.add_subplot(2, 1, 1)
    part2 = show.add_subplot(2, 1, 2)

    part1.imshow(convolved, cmap='gray')
    part1.set_title('Convolve2d')

    # Plotter real-delen av bildet.
    part2.imshow(picture_convolved.real, cmap='gray')
    part2.set_title('Fft2')

    plt.show()


plot_oppgave1_1()
