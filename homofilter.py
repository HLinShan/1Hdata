#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 11:18:17 2019

@author: smartdsp
"""

import numpy as np
import os
import cv2
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import csv

def cut(img):
    thresh = threshold_otsu(img)
#    print(thresh)
    binary = img > thresh
    minx, miny = 0, 0
    maxx, maxy = img.shape[0], img.shape[1]
    
    direction = np.sum(binary[img.shape[0]//2, 0:img.shape[1]//2])
    if direction < np.sum(binary[img.shape[0]//2, :])/2:
        maxy = img.shape[1]
        for yy in range(int(img.shape[1]-1), -1, -1):
            if sum(binary[:, yy]>0) == 0: 
                miny = yy
                break
    else:
        miny = 0
        for yy in range(0, img.shape[1], 1):
            if sum(binary[:, yy]>0) == 0: 
                maxy = yy
                break

    for xx in range(img.shape[0]):
      if sum(binary[xx, miny:maxy]!=0) > 0:
        minx = xx
        break
    for xx in range(img.shape[0]-1,0,-1):
      if sum(binary[xx, miny:maxy]!=0) > 0:
        maxx = xx
        break
#    print(minx, maxx, miny, maxy)
    img = img[minx:maxx+1, miny:maxy+1]
    return img, minx, maxx, miny, maxy
    
def homofilter(img, rH=2., rL=0.5, c=1.0, d0=20.):
    
    h, w = img.shape
    P = np.floor(h/2)
    Q = np.floor(w/2)
    
    img1 = np.log(img+1)
    F1 = np.fft.fft2(img1)
    F1 = np.fft.fftshift(F1)
    
    GH = np.zeros((h,w))
    GL = np.zeros((h,w))
    a = d0**2
    r = rH - rL
    for i in range(h):
        for j in range(w):
            temp = np.sqrt((i-P)**2+(j-Q)**2)
            GH[i,j] = r*(1-np.exp((-c)*(temp**2/a)))+rL
            GL[i,j] = np.exp(-0.5*(temp**2/a))
    
    GH = F1 * GH
    GH = np.fft.ifftshift(GH)
    hp = np.fft.ifft2(GH)
    
    GL = F1 * GL
    GL = np.fft.ifftshift(GL)
    lp = np.fft.ifft2(GL)
    
    hp = np.real(hp)
    lp = np.real(lp)
            
    imgh = np.exp(hp)-1
    imgh = 255*(imgh-np.min(imgh))/(np.max(imgh)-np.min(imgh))
    imgl = np.exp(lp)-1
    imgl = 255*(imgl-np.min(imgl))/(np.max(imgl)-np.min(imgl))
#    print(np.max(imgh), np.max(imgl))
    
    return imgh, imgl

def BilateralFilter(img):
    
    limg = img.copy()
    for i in range(20):
        limg = cv2.bilateralFilter(limg, 1, 140, 140)
#        plt.figure("limg"+str(i)) # 图像窗口名称
#        plt.imshow(limg)
#        plt.axis('off') # 关掉坐标轴为 off
#        plt.title('limg'+str(i)) # 图像题目
#        plt.show()
    himg = abs(img - limg)
    return himg, limg

def read_mias_csv(csvfile):
    
    with open(csvfile) as f:
        
        txt = open('./data/mias.txt', 'w')        
        reader = csv.DictReader(f)
        rows = [row for row in reader]
        imgname = [row['image_name'].strip() for row in rows[1:]]
        pathology = [row['pathology'].lower() for row in rows[1:]]
        centre_x = [row['centre_x'] for row in rows[1:]]
        centre_y = [row['centre_y'] for row in rows[1:]]
        radius = [row['radius'] for row in rows[1:]]
        for (name, patho, x, y, r) in zip(imgname, pathology, centre_x, centre_y, radius):
            
            if patho == 'benign':
                label = 0
            elif patho == 'malignant':
                label = 1
            else:
                label = 2
            txt.write(name+'.png '+str(label)+' '+x+' '+y+' '+r+'\n')
        txt.close()
        
        
if __name__ == '__main__':
#    read_mias_csv('./data/MIAS.csv')
    
#    DATA_DIR = './Inbreast_images'#所有图片放在这个文件夹里
#    TRAIN_IMAGE_LIST = './data/train_inbreast.txt'
#    VALID_IMAGE_LIST = './data/test_inbreast.txt'
    
#    DATA_DIR = './1Hdata/cutimage'
#    VALID_IMAGE_LIST = './1Hdata/full.txt'
    
#    DATA_DIR = '../end2end-all-conv/full'
#    TRAIN_IMAGE_LIST = '../end2end-all-conv/full/full_train.txt'
#    VALID_IMAGE_LIST = '../end2end-all-conv/full/full_val.txt'
#    HEATMAP_IMAGE_LIST = '../end2end-all-conv/full/full_test.txt'
    
    DATA_DIR = './Mias_images'
    VALID_IMAGE_LIST = './data/mias.txt'
    
    image_names = []
    labels = []
    centerx = []
    centery = []
    radius = []
#    with open(TRAIN_IMAGE_LIST, "r") as f:            
#        for line in f:
#            items = line.split()
#            image_name= items[0]
#            image_name = os.path.join(DATA_DIR, image_name)
#            image_names.append(image_name)
    with open(VALID_IMAGE_LIST, "r") as f:            
        for line in f:
            items = line.split()
            image_name= items[0]
            image_name = os.path.join(DATA_DIR, image_name)
            image_names.append(image_name)
#            label = items[1]
#            x = int(items[2])
#            y = int(items[3])
#            r = int(items[4])
#            labels.append(label)
#            centerx.append(x)
#            centery.append(y)
#            radius.append(r)
#    with open(HEATMAP_IMAGE_LIST, "r") as f:            
#        for line in f:
#            items = line.split()
#            image_name= items[0]
#            image_name = os.path.join(DATA_DIR, image_name)
#            image_names.append(image_name)
            
    for imgname in image_names:
#    for (imgname, label, x, y, r) in zip(image_names, labels, centerx, centery, radius):
        image = cv2.imread(imgname, cv2.IMREAD_UNCHANGED)
        height, width = image.shape
#        image = image / 65535. * 255.
#        cutimage, minx, maxx, miny, maxy = cut(image)
        cutimage = cv2.resize(image, (448, 576))
#        newx = (x - miny)*576./height
#        newy = (y - minx)*448./width
#        newr = r*((448.*576.)/(height*width))
#        mask = np.zeros(image.shape)
#        mask[np.where(image>0)] = 1
#        mask[np.where(image==0)] = 0
#        threshold = (np.sum(image)/np.sum(mask))
#        mask = np.zeros(image.shape)
#        mask[np.where(image>=threshold)] = 1
#        mask[np.where(image<threshold)] = 0
#        adpimg = image * mask
        
#        plt.figure("cutimage") # 图像窗口名称
#        plt.imshow(cutimage)
#        plt.axis('off') # 关掉坐标轴为 off
#        plt.title('cutimage') # 图像题目
#        plt.show()
#        plt.figure("adpimg") # 图像窗口名称
#        plt.imshow(adpimg)
#        plt.axis('off') # 关掉坐标轴为 off
#        plt.title('adpimg') # 图像题目
#        plt.show()
#        print(np.max(image))
        
        cutimage = cutimage.astype('float32')
        x = np.zeros(cutimage.shape + (3,), dtype='float32')
        x[:,:,0] = cutimage        
#        (x[:,:,1], x[:,:,2]) = homofilter(cutimage)
        (x[:,:,1], x[:,:,2]) = BilateralFilter(cutimage)
#        x[:,:,1] = adpimg
        cutimage = Image.fromarray(x[:,:,:].astype('uint8'))
        
        savename = 'BFHL_'+imgname.split('/')[-1].split('.')[0]+'.png'
#        cutimage.save(os.path.join(DATA_DIR, savename))
        
        image = Image.fromarray(x[:,:,0].astype('uint8'))
        savename = 'visualize0.png'
        image.save(savename)
        image = Image.fromarray(x[:,:,1].astype('uint8'))
        savename = 'visualize1.png'
        image.save(savename)
        image = Image.fromarray(x[:,:,2].astype('uint8'))
        savename = 'visualize2.png'
        image.save(savename)
        image = Image.fromarray(x.astype('uint8'))
        savename = 'visualize.png'
        image.save(savename)
        
        plt.figure("before") # 图像窗口名称
        plt.imshow(cutimage)
        plt.axis('off') # 关掉坐标轴为 off
        plt.title('before') # 图像题目
        plt.show()
        a=0