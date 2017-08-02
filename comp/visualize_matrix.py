#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 20:32:06 2017

@author: yutingzhao
"""

'''==========================Plot Confusion Matrix========================='''
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import scipy.io as scio

pcanet = scio.loadmat("/home/yutingzhao/CodeDemo/TentNet-Tsingzao/result/pcanet_uiuc.mat")
tennet = scio.loadmat("/home/yutingzhao/CodeDemo/TentNet-Tsingzao/result/tennet_uiuc.mat")

org_true = pcanet['TestLabels']
org_pre  = pcanet['predict_label']
sgm_true = tennet['TestLabel']
sgm_pre  = tennet['predict_label']

cn_matrix = []
cn_matrix.append(confusion_matrix(org_true, org_pre))
cn_matrix.append(confusion_matrix(sgm_true, sgm_pre))
all_title = ['PCANet Confusion Matrix', 'TenneT Confusion Matrix']
classes = ['celltoear','drink','getup','point','running','throw','walk']

fig, axes = plt.subplots(nrows=1, ncols=2)
for ax,title,cnf_matrix in zip(axes.flat,all_title,cn_matrix):
    im = ax.imshow(cnf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=30)
    ax.set_yticks(np.arange(len(classes)))
    ax.set_yticklabels(classes)

    thresh = cnf_matrix.max() / 2.
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        ax.text(j, i, cnf_matrix[i, j],
                 horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] < 100 else "black")
    
fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.69)

plt.show()

'''===================Plot Accuracy v.s. Class=============================='''
 
import numpy as np
import matplotlib.pyplot as plt


n_groups = 9

penn2 = np.array((52.84,52.18,71.40,71.62,68.34,72.05,64.63,67.90,75.45))
penn5 = np.array((28.65,28.21,44.57,42.92,42.15,46.71,57.03,44.29,69.65))
penn8 = np.array((20.34,20.30,39.52,35.61,36.06,41.13,51.73,35.65,61.07))
penn11= np.array((14.83,14.24,31.95,30.75,26.32,33.02,42.38,27.03,55.07))
penn15= np.array((10.66,11.13,25.76,26.25,21.62,28.89,37.05,25.19,52.11))

fig, ax = plt.subplots(figsize=(16,5))

index = np.arange(n_groups*5)[::5]+1
bar_width = 0.8

opacity = 0.4

rects1 = plt.bar(index, penn2, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Penn - 2 classes')

rects2 = plt.bar(index + bar_width, penn5, bar_width,
                 alpha=opacity,
                 color='r',
                 label='Penn - 5 classes')

rects3 = plt.bar(index + bar_width*2, penn8, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Penn - 8 classes')

rects4 = plt.bar(index + bar_width*3, penn11, bar_width,
                 alpha=opacity,
                 color='m',
                 label='Penn - 11 classes')

rects5 = plt.bar(index + bar_width*4, penn15, bar_width,
                 alpha=opacity,
                 color='y',
                 label='Penn - 15 classes')

plt.plot(np.ones(46)*75.45, color='b', linestyle='--', linewidth='2')
plt.plot(np.ones(46)*69.65, color='r', linestyle='--', linewidth='2')
plt.plot(np.ones(46)*61.07, color='g', linestyle='--', linewidth='2')
plt.plot(np.ones(46)*55.07, color='m', linestyle='--', linewidth='2')
plt.plot(np.ones(46)*52.11, color='y', linestyle='--', linewidth='2')

plt.ylabel('Testing Accuracy')
plt.ylim(0,100)  
plt.xticks(index + bar_width*3, ('AveragePooling+SVM', 'MaxPooling+SVM', 'LeNet', 'CIFARNet', 
'FCNet', '3DNet', 'PCANet', 'C3D*', 'TenneT'))
#==============================================================================
# legend = plt.legend(bbox_to_anchor=(1.5, 1),ncol=5)
# legend.get_title().set_fontsize(fontsize = 'small')
#==============================================================================

plt.legend(bbox_to_anchor=(1, 1),ncol=5)
leg = plt.gca().get_legend()
ltext  = leg.get_texts()
plt.setp(ltext, fontsize='small')

plt.tight_layout()
plt.show()   

'''==========================Accuracy v.s. Resolution======================='''

import numpy as np
import matplotlib.pyplot as plt

pcanet = np.array((59.44,57.03,48.96,37.16))
tennet = np.array((69.97,69.65,66.14,50.88))

plt.figure(figsize=(10,5))
plt.plot(pcanet, marker='*', linestyle='--', color='r', markersize=10)
plt.plot(tennet, marker='^', linestyle='-.', color='b', markersize=8)

plt.ylabel('Testing Accuracy')
plt.ylim(20,80)  
plt.xlabel('Video Resolution')
plt.xticks((0,1,2,3),('64x64','32x32','16x16','8x8'))
plt.grid()
plt.legend(['PCANet','TenneT'])

'''==========================Visualization of Kernels======================='''
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt

data  = scio.loadmat('/home/yutingzhao/CodeDemo/TentNet-Tsingzao/result/visual.mat')
data0 = np.reshape(data['V'][0][0],(3,3,3,10))
data1 = np.reshape(data['V'][1][0],(3,3,3,10))
data  = [data0,data1]

for j in range(2):
    plt.figure(figsize=(6,2))
    for i in range(10):    
        ax = plt.subplot(2, 5, i+1)
        plt.imshow(data[j][:,:,0,i])
        ax.get_xaxis().set_visible(False) 
        ax.get_yaxis().set_visible(False)

'''==========================Visualization of Videos======================='''
import matplotlib.pyplot as plt 
from PIL import Image
import numpy as np
import random

img_path = '/home/yutingzhao/train.lst'
fp = open(img_path, 'r')
lines = fp.readlines()
fp.close()
img_numb = len(lines)

plt.figure(figsize=(9,5))
ind = [random.randint(0,img_numb) for _ in range(8)]
       
for i in range(8):
    frame_path = lines[ind[i]].strip().split(' ')[0]+'/frame-'+lines[ind[i]].strip().split(' ')[1].zfill(4)+'.jpg'
    img = Image.open(frame_path)
    for j in range(4):
        ax = plt.subplot(4,8,8*j+i+1)
        temp = img.resize((2**(6-j),2**(6-j)))
        plt.imshow(temp)
        plt.axis('off')
    

  







