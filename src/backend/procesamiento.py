import sys
import os

import tensorflow as tf
from tensorflow.keras import * 
from tensorflow.keras.layers import *

import numpy as np
from numpy import random
from random import randint
from glob import glob

from skimage import io
from skimage.draw import circle
from skimage.filters import sobel, sato, threshold_li, frangi
from skimage import morphology

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from IPython.display import SVG

from PIL import Image
import cv2
import matplotlib.pyplot as plt



def Segmentation_model_2():
    inputs = Input((None, None, 3))
    # Encoder
    block_1 = Dropout(0.1)(ReLU()(Conv2D(32, (3, 3), padding='same')(inputs)))
    block_1 = Dropout(0.1)(ReLU()(Conv2D(32, (3, 3), padding='same')(block_1)))
    down_1 = MaxPooling2D(pool_size=(2, 2))(block_1) #32

    block_2 = Dropout(0.1)(ReLU()(Conv2D(64, (3, 3), padding='same')(down_1)))
    block_2 = Dropout(0.1)(ReLU()(Conv2D(64, (3, 3), padding='same')(block_2)))
    down_2 = MaxPooling2D(pool_size=(2, 2))(block_2) #16

    block_3 = Dropout(0.1)(ReLU()(Conv2D(128, (3, 3), padding='same')(down_2)))
    block_3 = Dropout(0.1)(ReLU()(Conv2D(128, (3, 3), padding='same')(block_3)))
    down_3 = MaxPooling2D(pool_size=(2, 2))(block_3) #8

    block_4 = Dropout(0.1)(ReLU()(Conv2D(256, (3, 3), padding='same')(down_3)))
    block_4 = Dropout(0.1)(ReLU()(Conv2D(256, (3, 3), padding='same')(block_4)))
    down_4 = MaxPooling2D(pool_size=(2, 2))(block_4) #4
    
    # Cuello de botella
    bottle_neck = Dropout(0.1)(ReLU()(Conv2D(512, (3, 3), padding='same')(down_4)))
    bottle_neck = Dropout(0.1)(ReLU()(Conv2D(512, (3, 3), padding='same')(bottle_neck)))

    # Decoder
    up_1 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(bottle_neck), block_4], axis=3) #8
    block_5 = Dropout(0.1)(ReLU()(Conv2D(256, (3, 3), padding='same')(up_1)))
    block_5 = Dropout(0.1)(ReLU()(Conv2D(256, (3, 3), padding='same')(block_5)))

    up_2 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(block_5), block_3], axis=3) #16
    block_6 = Dropout(0.1)(ReLU()(Conv2D(128, (3, 3), padding='same')(up_2)))
    block_6 = Dropout(0.1)(ReLU()(Conv2D(128, (3, 3), padding='same')(block_6)))

    up_3 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(block_6), block_2], axis=3) #32
    block_7 = Dropout(0.1)(ReLU()(Conv2D(64, (3, 3), padding='same')(up_3)))
    block_7 = Dropout(0.1)(ReLU()(Conv2D(64, (3, 3), padding='same')(block_7)))

    up_4 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(block_7), block_1], axis=3) #64
    block_8 = Dropout(0.1)(ReLU()(Conv2D(32, (3, 3), padding='same')(up_4)))
    block_8 = Dropout(0.1)(ReLU()(Conv2D(32, (3, 3), padding='same')(block_8)))

    output = Conv2D(1, (1, 1), activation='sigmoid')(block_8)

    model = Model(inputs=[inputs], outputs=[output])

    return model


def obtenerImagenes(filename, wPath, tmpPath):
    # Redes Convolucionales
    Seg_model = Segmentation_model_2()
    Seg_model.load_weights(wPath, by_name=True)

    in_img = plt.imread(filename, 3) / 255
    in_img = tf.image.resize(in_img, (64*9, 64*9))
    n_img = tf.image.resize(in_img, (64*9, 64*9))
    in_img = tf.stack([in_img, n_img], axis=0)

    salida = Seg_model(in_img, training = False)

    res = salida[0,...]
    res = tf.image.resize(res, (584, 565))
    res = res.numpy()
    res = res.reshape(584,565)

    binary = res > .1
    binary_clean = morphology.remove_small_objects(binary, 3000)
    prueba = np.clip(res,0 , 1)

    cv2.imwrite(tmpPath + "/deepSeg.png", binary_clean * 255)

    # Filtro de Sato

    image = cv2.imread(filename, 0)

    ring = np.zeros((584,565))
    rr, cc = circle(292, 282, 250, shape=(image.shape))
    ring[rr, cc] = 1 


    elevation_map = sobel(image)
    satoi = sato(image)

    thresh = threshold_li(satoi)
    binary = satoi > thresh
    frgi = frangi(image)
    threshF = threshold_li(frgi)

    bF = frgi * 100000 > .03
    bF = morphology.remove_small_objects(bF, 3000)
    bF = bF * ring  #imagen segmentada 1

    binary_clean = morphology.remove_small_objects(binary, 3000)

    l_binary_clean = morphology.label(binary_clean, return_num=True, connectivity=1)

    binary_f = binary_clean * ring # segmentada 2, solo arterias principales

    cv2.imwrite(tmpPath + "/SatoSeg.png", bF * 255)

    shape = binary_f.shape
    a = shape[0]/2 - 1
    b = shape[1] + 1 /2
    centro_ = (233, 291)

    x_1 = y_2 = y_3 = px = i = lado = 0
    y_1 = a
    #primer pixel blanco centro
    while px != 1:
        px = binary_f[int(a)][i]
        i = i + 1
    x_1 = i

    if x_1 > shape[1] / 2:
        lado = 0
        i = shape[1] - 1
        px = 0
        while px != 1:
            px = binary_f[int(a)][i]
            i = i - 1
    else:
        lado = 1

    if lado == 0:
        b = (shape[1] + 1) / 4
        b = (b * 3) - 20
    else:
        b = ((shape[1] + 1) / 4) + 20 
    x_1 = i
    x_2 = b
    x_3 = x_2


    px, i = [0, 0]
    #primer pixel blanco arriba
    while px != 1:
        px = binary_f[i][int(b)]
        i = i + 1
    y_2 = i
    px, i = [0, shape[0] - 1]
    #primer pixel blanco abajo
    while px != 1:
        px = binary_f[i][int(b)]
        i = i - 1
    y_3 = i

    if x_1 < shape[1] / 2:
        lado = 1

    punto1 = (int(x_1), int(y_1))
    punto2 = (int(x_2), int(y_2))
    punto3 = (int(x_3), int(y_3))
    centro = [b, a]

    color = (0, 0, 255)
    # Para calculo de angulos Tan(x) = CatetoOpuesto / CatetoAdyacente
    ca =  np.abs(centro[0] - x_1)
    co1 = np.abs(centro[1] - y_2)
    co2 = np.abs(centro[1] - y_3)

    ca_ = (int(centro[0]), int(y_1)) # coordernadas para impresion
    co_ = (int(centro[0]), y_2) # coordernadas para impresion
    
    img2 = (binary_f * 255) + image
    l = cv2.line(img2, punto1, punto2, color, 5)
    l = cv2.line(l, punto1, punto3, color, 5)
    l = cv2.line(l, punto1, ca_, color, 2)
    l = cv2.line(l, ca_, co_, color, 2)
    l = cv2.circle(l, punto1, 30, color, 2)


    angulo1 = np.rad2deg(np.arctan(co1 / ca)) # x = arcTan((Co/Ca))
    angulo2 = np.rad2deg(np.arctan(co2 / ca)) 

    angulo = np.abs(angulo1) + np.abs(angulo2)
    cv2.putText(l, "{0:.7}".format(angulo), centro_, cv2.QT_FONT_NORMAL, 0.8, (0,0,0), 2, cv2.LINE_AA)

    cv2.imwrite(tmpPath + "/lineas.png", l)


if __name__ == "__main__":
    st = sys.argv[1]
    st2 = sys.argv[2]
    st3 = sys.argv[3]
    obtenerImagenes(st, st2, st3)


