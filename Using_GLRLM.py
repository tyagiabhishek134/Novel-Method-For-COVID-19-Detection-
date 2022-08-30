# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 21:16:39 2022

@author: abcd
"""
import numpy as np 
from glrlm import GLRLM
import cv2
import matplotlib.pyplot as plt
import glob
import os

print(os.listdir("E:\Dataset_1\split 9010"))

#Resize images to
SIZE = 128

#Capture images and labels into arrays.
#Start by creating empty lists.
train_images = []
train_labels = [] 
#for directory_path in glob.glob("cell_images/train/*"):
for directory_path in glob.glob("E:/Dataset_2/split 9010/train/*"):
    label = directory_path.split("\\")[-1]
    print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        print(img_path)
        img = cv2.imread(img_path, 0) #Reading color images
        img = cv2.resize(img, (SIZE, SIZE)) #Resize images
        train_images.append(img)
        train_labels.append(label)
        
train_images = np.array(train_images)
train_labels = np.array(train_labels)

#Do exactly the same for test/validation images
# test
test_images = []
test_labels = []
#for directory_path in glob.glob("cell_images/test/*"): 
for directory_path in glob.glob("E:/Dataset_2/split 9010/test/*"):
    fruit_label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        img = cv2.imread(img_path, 0)
        img = cv2.resize(img, (SIZE, SIZE))
        test_images.append(img)
        test_labels.append(fruit_label)
        
test_images = np.array(test_images)
test_labels = np.array(test_labels)

#Encode labels from text (folder names) to integers.
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(test_labels)
test_labels_encoded = le.transform(test_labels)
le.fit(train_labels)
train_labels_encoded = le.transform(train_labels)

#Split data into test and train datasets (already split but assigning to meaningful convention)
#If you only have one dataset then split here
x_train, y_train, x_test, y_test = train_images, train_labels_encoded, test_images, test_labels_encoded

def glrlm_0(f, mask, grayLevel=5, runLength=5, skipFirstRow=True):
    degree0Matrix = np.zeros([grayLevel, runLength])
    counter = 0
    for y in range(f.shape[0]):
        for x in range(f.shape[1]):   
            nowVal = f[y][x]
            if x + 1 >= f.shape[1]:
                nextVal = None
            else:
                nextVal = f[y][x + 1]
            if nextVal != nowVal and counter == 0:
                degree0Matrix[int(nowVal)][counter] += 1
            elif nextVal == nowVal:
                counter += 1
            elif nextVal != nowVal and counter != 0:
                degree0Matrix[int(nowVal)][counter] += 1
                counter = 0
    return degree0Matrix[1:,:] if skipFirstRow else degree0Matrix

def glrlm_90(f, mask, grayLevel=5, runLength=5, skipFirstRow=True):
    degree90Matrix = np.zeros([grayLevel, runLength])
    counter = 0
    for x in range(f.shape[1]):
        for y in range(f.shape[0]):
            nowVal = f[y][x]
            if y + 1 >= f.shape[0]:
                nextVal = None
            else:
                nextVal = f[y + 1][x]
            if nextVal != nowVal and counter == 0:
                degree90Matrix[int(nowVal)][counter] += 1
            elif nextVal == nowVal:
                counter += 1
            elif nextVal != nowVal and counter != 0:
                degree90Matrix[int(nowVal)][counter] += 1
                counter = 0
    return degree90Matrix[1:,:] if skipFirstRow else degree90Matrix

def glrlm_45(f, mask, grayLevel=5, runLength=5, skipFirstRow=True):
    degree45Matrix = np.zeros([grayLevel, runLength])
    for y in range(f.shape[0]):
        counter = 0
        i_range = max(f.shape)
        for i in range(i_range):
            y1 = y - i
            if i >= f.shape[1] or y1 < 0:
                break
            else:
                nowVal = f[y1][i]
            if y1 - 1 < 0 or i + 1 >= f.shape[1]:
                nextVal = None
            else:
                nextVal = f[y1 - 1][i + 1]
            if nextVal != nowVal and counter == 0:
                degree45Matrix[int(nowVal)][counter] += 1
            elif nextVal == nowVal:
                counter += 1
            elif nextVal != nowVal and counter != 0:
                degree45Matrix[int(nowVal)][counter] += 1
                counter = 0
    for x in range(f.shape[1]):
        if x == f.shape[1] - 1:
            break
        counter = 0
        i_range = max(f.shape)
        for i in range(i_range):
            y_i = -1 - i
            x_i = -1 + i - x
            if x_i >= 0 or y_i <= -1 - f.shape[0]:
                break
            else:
                nowVal = f[y_i][x_i]
            if y_i - 1 <= -(f.shape[0] + 1) or x_i + 1 >= 0:
                nextVal = None
            else:
                nextVal = f[y_i - 1][x_i + 1]
            if nextVal != nowVal and counter == 0:
                degree45Matrix[int(nowVal)][counter] += 1
            elif nextVal == nowVal:
                counter += 1
            elif nextVal != nowVal and counter != 0:
                degree45Matrix[int(nowVal)][counter] += 1
                counter = 0
    degree45Matrix[0,1:] = 0
    return degree45Matrix[1:,:] if skipFirstRow else degree45Matrix

def glrlm_135(f, mask, grayLevel=5, runLength=5, skipFirstRow=True):
    degree135Matrix = np.zeros([grayLevel, runLength])
    for y in range(f.shape[0]):
        counter = 0
        i_range = max(f.shape)
        for i in range(i_range):
            y1 = y + i
            if y1 >= f.shape[0] or i >= f.shape[1]:
                break
            else:
                nowVal = f[y1][i]
                if y1 >= f.shape[0] - 1 or i >= f.shape[1] - 1:
                    nextVal = None
                else:
                    nextVal = f[y1 + 1][i + 1]
                if nextVal != nowVal and counter == 0:
                    degree135Matrix[int(nowVal)][counter] += 1
                elif nextVal == nowVal:
                    counter += 1
                elif nextVal != nowVal and counter != 0:
                    degree135Matrix[int(nowVal)][counter] += 1
                    counter = 0
    for x in range(f.shape[1]):
        if x == 0:
            continue
        i_range = max(f.shape)
        counter = 0
        for i in range(i_range):
            x1 = x + i
            if i >= f.shape[0] or x1 >= f.shape[1]:
                break
            else:
                nowVal = f[i][x1]
            if i >= f.shape[0] - 1 or x1 >= f.shape[1] - 1:
                nextVal = None
            else:
                nextVal = f[i + 1][x1 + 1]
            if nextVal != nowVal and counter == 0:
                degree135Matrix[int(nowVal)][counter] += 1
            elif nextVal == nowVal:
                counter += 1
            elif nextVal != nowVal and counter != 0:
                degree135Matrix[int(nowVal)][counter] += 1
                counter = 0
    degree135Matrix[0,1:] = 0
    return degree135Matrix[1:,:] if skipFirstRow else degree135Matrix

def _apply_over_degree(function, x1, x2):
    if function == np.divide:
        x2 = x2 + 1e-16
    rows, cols, nums = x1.shape
    result = np.ndarray((rows, cols, nums))
    for i in range(nums):
        result[:, :, i] = function(x1[:, :, i], x2)
        result[result == np.inf] = 0
        result[np.isnan(result)] = 0
    return result 
    
def _calculate_ij (rlmatrix):
    gray_level, run_length, _ = rlmatrix.shape
    I, J = np.ogrid[0:gray_level, 0:run_length]
    return I, J+1
    
def _calculate_s(rlmatrix):
    return np.apply_over_axes(np.sum, rlmatrix, axes=(0, 1))[0, 0]

def glrlm(f, mask, Ng=256):   
    '''
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    mask : numpy ndarray
        Mask image N1 x N2 with 1 if pixels belongs to ROI, 0 else.
    Ng : int, optional
        Image number of gray values. The default is 256.
    Returns
    -------
    mat : numpy ndarray
        GLRL Matrices for 0, 45, 90 and 135 degrees.
    '''
    runLength = max(f.shape)
    mat0 = glrlm_0(f, mask, grayLevel=Ng, runLength=runLength)
    mat45 = glrlm_45(f, mask, grayLevel=Ng, runLength=runLength)
    mat90 = glrlm_90(f, mask, grayLevel=Ng, runLength=runLength)
    mat135 = glrlm_135(f, mask, grayLevel=Ng, runLength=runLength)            
    mat = np.dstack((mat0, mat45, mat90, mat135))      
    return mat

def glrlm_features(f, mask, Ng=256):
    '''
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    mask : numpy ndarray
        Mask image N1 x N2 with 1 if pixels belongs to ROI, 0 else. Give None
        if you want to consider ROI the whole image.
    Ng : int, optional
        Image number of gray values. The default is 256.
    Returns
    -------
    features : numpy ndarray
        1)Short Run Emphasis, 2)Long Run Emphasis, 3)Gray Level 
        Non-Uniformity/Gray Level Distribution, 4)Run Length 
        Non-Uniformity/Run Length Distribution, 5)Run Percentage,
        6)Low Gray Level Run Emphasis, 7)High Gray Level Run Emphasis,
        8)Short Low Gray Level Emphasis, 9)Short Run High Gray Level 
        Emphasis, 10)Long Run Low Gray Level Emphasis, 11)Long Run 
        High Gray Level Emphasis.
    labels : list
        Labels of features.
    '''
    
    if mask is None:
        mask = np.ones(f.shape)
        
    labels = ["GLRLM_ShortRunEmphasis",
              "GLRLM_LongRunEmphasis",
              "GLRLM_GrayLevelNo-Uniformity",
              "GLRLM_RunLengthNonUniformity",
              "GLRLM_RunPercentage",
              "GLRLM_LowGrayLevelRunEmphasis",
              "GLRLM_HighGrayLevelRunEmphasis",
              "GLRLM_Short owGrayLevelEmphasis",
              "GLRLM_ShortRunHighGrayLevelEmphasis",
              "GLRLM_LongRunLowGrayLevelEmphasis",
              "GLRLM_LongRunHighGrayLevelEmphasis"]
    
    rlmatrix = glrlm(f, mask, Ng)
        
    I, J = _calculate_ij(rlmatrix)
    S = _calculate_s(rlmatrix)
    G = np.apply_over_axes(np.sum, rlmatrix, axes=1)
    R = np.apply_over_axes(np.sum, rlmatrix, axes=0)
        
    features = np.zeros(11,np.double)
    features[0] = ((np.apply_over_axes(np.sum, _apply_over_degree(np.divide, rlmatrix, (J*J)), axes=(0, 1))[0, 0])/S).mean()
    features[1] = ((np.apply_over_axes(np.sum, _apply_over_degree(np.multiply, rlmatrix, (J*J)), axes=(0, 1))[0, 0])/S).mean()
    features[2] = ((np.apply_over_axes(np.sum, (G*G), axes=(0, 1))[0, 0])/S).mean()
    features[3] = ((np.apply_over_axes(np.sum, (R*R), axes=(0, 1))[0, 0])/S).mean()
        
    gray_level, run_length,_ = rlmatrix.shape
    num_voxels = gray_level * run_length
    features[4] = (S/num_voxels).mean()
        
    features[5]= ((np.apply_over_axes(np.sum, _apply_over_degree(np.divide, rlmatrix, (I*I)), axes=(0, 1))[0, 0])/S).mean()
    features[6] = ((np.apply_over_axes(np.sum, _apply_over_degree(np.multiply, rlmatrix, (I*I)), axes=(0, 1))[0, 0])/S).mean()
    features[7] = ((np.apply_over_axes(np.sum, _apply_over_degree(np.divide, rlmatrix, (I*I*J*J)), axes=(0, 1))[0, 0])/S).mean()
        
    temp = _apply_over_degree(np.multiply, rlmatrix, (I*I))
    features[8] = ((np.apply_over_axes(np.sum, _apply_over_degree(np.divide, temp, (J*J)), axes=(0, 1))[0, 0])/S).mean()
        
    temp = _apply_over_degree(np.multiply, rlmatrix, (J*J))
    features[9] = ((np.apply_over_axes(np.sum, _apply_over_degree(np.divide, temp, (J*J)), axes=(0, 1))[0, 0])/S).mean()
    features[10] = ((np.apply_over_axes(np.sum, _apply_over_degree(np.multiply, rlmatrix, (I*I*J*J)), axes=(0, 1))[0, 0])/S).mean()
        
    return features, labels    