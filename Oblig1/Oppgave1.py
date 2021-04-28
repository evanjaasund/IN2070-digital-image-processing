from imageio import imread
import matplotlib.pyplot as plt
import numpy as np
import math

def histo(picture):
    N,M = picture.shape
    h = np.zeros(256)

    for x in range(N):
        for y in range(M):
            if int(picture[x,y]) < len(h):
                h[int(picture[x,y])] += 1
            else:
                h[255] += 1
    return h

def normHisto(histogram):
    normHist = np.zeros(256)
    totalP = sum(histogram)

    for i in range(len(histogram)):
        normHist[i] = (histogram[i])/totalP
    return normHist


def find_mean(normHistogram):
    my = 0

    for i in range(len(normHistogram)):
        my += i*normHistogram[i]
    return my

def find_standard_diviation(normHistogram):
    mean = find_mean(normHistogram)
    variance = 0

    for i in range(len(normHistogram)):
        variance += i**2*normHistogram[i]
    return math.sqrt(int(variance-mean**2))

def standard_diviation_mean_change(std_div, mean, picture):
    N,M = picture.shape

    picture_out = np.zeros((N,M))
    histogram_ = histo(picture)
    normHistogram_ = normHisto(histogram_)

    a = std_div/find_standard_diviation(normHistogram_)
    b = mean - (a*find_mean(normHistogram_))

    for x in range(N):
        for y in range(M):
            picture_out[x][y] = a*picture[x][y]+b
    return picture_out


def bilinear_interpolasjon(x, y, picture):
  x_max = np.shape(picture)[0]
  y_max = np.shape(picture)[1]

  x_0 = int(np.floor(x))
  y_0 = int(np.floor(y))
  x_1 = int(np.ceil(x))
  y_1 = int(np.ceil(y))

  delta_x = x - x_0
  delta_y = y - y_0
  
  if x_0 >= 0 and x_1 < x_max and y_0 >= 0 and y_1 < y_max:
    p = picture[x_0][y_0] + ((picture[x_1][y_0]-picture[x_0][y_0]))*delta_x
    q = picture[x_0][y_1] + ((picture[x_1][y_1]-picture[x_0][y_1]))*delta_x
    return p + (q - p)*delta_y
  else:
    return 0
    

def neighbour_interpolasjon(x, y, picture):
    x_max, y_max = picture.shape
  
    x_new = int(np.round(x))
    y_new = int(np.round(y))

    if x < x_max and y < y_max:
        return picture[x_new][y_new]
    else:
        return 0


#Redigert forlengsmapping:
def forward_mapping(picture, f_out, transform_matrix):
    N,M = picture.shape
    G,H = f_out.shape

    f_out = np.zeros((G,H))

    for x_in in range(N):
        for y_in in range(M):
            x_y_out = np.dot(transform_matrix, [x_in, y_in, 1])
            if int(np.round(x_y_out[0])) < G and int(np.round(x_y_out[1])) < H:
                f_out[int(np.round(x_y_out[0]))][int(np.round(x_y_out[1]))] = picture[x_in][y_in]
            
    return f_out


def backwards_mapping(picture, f_out_shape, transform_matrix, interpolasjon_type = None):
    N,M = picture.shape
    G,H = f_out_shape.shape
    g_out = np.zeros((G, H))
    inv_transform = np.linalg.inv(transform_matrix)
    for x_in in range(G):
        for y_in in range(H):
            v_w_out = np.dot(inv_transform, [x_in, y_in, 1])
            
            if v_w_out[0] < N and v_w_out[1] < M and interpolasjon_type in ["bilinear", None]:
                g_out[x_in][y_in] = bilinear_interpolasjon(v_w_out[0], v_w_out[1], picture)
            else:
                if v_w_out[0] < N and v_w_out[1] < M and interpolasjon_type == "neighbour":
                    g_out[x_in][y_in] = neighbour_interpolasjon(v_w_out[0], v_w_out[1], picture)
    return g_out


def fixedProfile():
    G = np.array([[88, 84, 1],[68, 120, 1],[93, 118, 1],[116, 116, 1],[100, 142, 1]])
    G_T = G.transpose()
    G_T_G = np.linalg.inv(G_T@G)

    d_x = np.array([[169], [342], [256], [193], [316]])
    d_y = np.array([[258], [259], [376], [440], [441]])


    a = (G_T_G)@(G_T)@d_y 
    b = (G_T_G)@(G_T)@d_x

    trans_matrix = np.array([a.transpose()[0], b.transpose()[0], [0, 0, 1]])
    return trans_matrix



left_eye = 169, 258
right_eye = 342, 259
nose_tip = 256, 376
left_mouth = 193, 440
right_mouth = 316, 441

silvester_lEye = 84, 88
silvester_rEye = 120 , 68
silvester_nTip = 118, 93
silvester_lMouth = 116, 116
silvester_rMouth = 142, 100


picture = imread('portrett.png', as_gray=True)
org_histogram = histo(picture)
norm_org_histogram = normHisto(org_histogram)

new_picture = standard_diviation_mean_change(64, 127, picture)
new_histogram = histo(new_picture)
new_norm_histogram = normHisto(new_histogram)

geometrimaske = imread('geometrimaske.png', as_gray=True)
show = plt.figure()


part1 = show.add_subplot(2, 2, 1)
part2 = show.add_subplot(2, 2, 2)
part3 = show.add_subplot(2, 2, 3)
part4 = show.add_subplot(2, 2, 4)


fixed = fixedProfile()

fixed_forward = forward_mapping(new_picture, geometrimaske, fixed)
fixed_backwards_bilinear = backwards_mapping(new_picture, geometrimaske, fixed)
fixed_backwards_neighbour = backwards_mapping(new_picture, geometrimaske, fixed, interpolasjon_type = 'neighbour')

part1.imshow(new_picture, cmap='gray', vmin = 0, vmax = 255)
part1.set_title('Greyscale transformation')

part2.imshow(fixed_forward, cmap='gray', vmin = 0, vmax = 255)
part2.set_title('Fixed forward')


part3.imshow(fixed_backwards_neighbour, cmap='gray', vmin = 0, vmax = 255)
part3.set_title('Fixed backwards w/ neighbour')

part4.imshow(fixed_backwards_bilinear, cmap='gray', vmin = 0, vmax = 255)
part4.set_title('Fixed backwards w/ bilinear')

plt.show()

