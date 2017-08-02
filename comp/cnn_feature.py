#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 18:50:28 2017
TENNET
@author: yutingzhao
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.system('echo $CUDA_VISIBLE_DEVICES')

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.8
set_session(tf.Session(config=config))

from keras.layers import Input, Conv2D, Dense, MaxPooling2D, Flatten, Dropout, GlobalAveragePooling2D, Conv3D, MaxPooling3D
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.utils.np_utils import to_categorical

import scipy.io as scio
import random
import numpy as np
import h5py
    
'''=============================Construct Network=========================='''
def lenet(num_class):
    input_frame = Input((28,28,1))
    x = Conv2D(32, (3, 3), activation='relu')(input_frame)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    y = Dense(num_class, activation='softmax')(x)
    return Model(input_frame, y)

def cifarnet(num_class):
    input_frame = Input((32,32,1))
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(input_frame)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.25)(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(input_frame)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    y = Dense(num_class, activation='softmax')(x)
    return Model(input_frame, y)
 
def imagenet(num_class):
    base_model = InceptionV3(weights='imagenet')
    x = GlobalAveragePooling2D()(base_model.get_layer('mixed10').output)
    y = Dense(num_class, activation='softmax')(x)
    return Model(base_model.input, y)
    
def fcnet(num_class):
    input_frame = Input((1024,))
    x = Dense(512, activation='relu')(input_frame)
    x = Dropout(0.5)(x)
    y = Dense(num_class, activation='softmax')(x)
    return Model(input_frame, y)
    
def c3dnet(num_class):
    input_frame = Input((8,8,16,1))
    x = Conv3D(32, (3, 3, 3), padding='same', activation='relu')(input_frame)
    x = Conv3D(32, (3, 3, 3), activation='relu')(x)
    x = MaxPooling3D((2, 2, 1))(x)
    x = Dropout(0.25)(x)
    x = Conv3D(64, (3, 3, 3), padding='same', activation='relu')(input_frame)
    x = Conv3D(64, (3, 3, 3), activation='relu')(x)
    x = MaxPooling3D((2, 2, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    y = Dense(num_class, activation='softmax')(x)
    return Model(input_frame, y)

def tentnet(num_class):
    input_frame = Input((8,8,16,1)) 
    x = Conv3D(10, (3, 3, 3), padding='same', activation='relu')(input_frame)
    x = Conv3D(10, (3, 3, 3), activation='relu')(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    y = Dense(num_class, activation='softmax')(x)
    return Model(input_frame, y)   
   
'''=========================Data Releated=================================='''
def load_data(data_name, nb_class, pool='random'):
    train_file = '/home/yutingzhao/CodeDemo/TentNet-Tsingzao/data/'+data_name+'_train_'+str(nb_class)+'.mat'
    test_file  = '/home/yutingzhao/CodeDemo/TentNet-Tsingzao/data/'+data_name+'_test_'+str(nb_class)+'.mat'
    if pool=='average':
        temp = scio.loadmat(train_file)
        train_data = np.asarray(np.transpose(np.reshape(np.mean(temp['feature'],-1),(32,32,-1)),(2,1,0)),dtype='float32')
        train_label= temp['label']
        temp = scio.loadmat(test_file)
        test_data  = np.asarray(np.transpose(np.reshape(np.mean(temp['feature'],-1),(32,32,-1)),(2,1,0)),dtype='float32')
        test_label = temp['label']
    elif pool=='max':
        temp = scio.loadmat(train_file)
        train_data = np.asarray(np.transpose(np.reshape(np.max(temp['feature'],-1),(32,32,-1)),(2,1,0)),dtype='float32')
        train_label= temp['label']
        temp = scio.loadmat(test_file)
        test_data  = np.asarray(np.transpose(np.reshape(np.max(temp['feature'],-1),(32,32,-1)),(2,1,0)),dtype='float32')
        test_label = temp['label']
    elif pool=='none':
        temp = scio.loadmat(train_file)
        train_data = np.asarray(np.transpose(np.reshape(temp['feature'],(16,16,-1,16,1)),(2,1,0,3,4)),dtype='float32')
        train_label= temp['label']
        temp = scio.loadmat(test_file)
        test_data  = np.asarray(np.transpose(np.reshape(temp['feature'],(16,16,-1,16,1)),(2,1,0,3,4)),dtype='float32')
        test_label = temp['label']
    else:
        temp = scio.loadmat(train_file)
        train_data = np.asarray(np.transpose(temp['feature'][:,:,random.randint(0,15)].reshape(32,32,-1),(2,1,0)),dtype='float32')
        train_label= temp['label']
        temp = scio.loadmat(test_file)
        test_data  = np.asarray(np.transpose(temp['feature'][:,:,random.randint(0,15)].reshape(32,32,-1),(2,1,0)),dtype='float32')
        test_label = temp['label']
    return train_data, train_label, test_data, test_label

def load_data(data_name, nb_class, pool='random'):
    train_file = '/home/yutingzhao/CodeDemo/TentNet-Tsingzao/data/'+data_name+'_train_'+str(nb_class)+'.mat'
    test_file  = '/home/yutingzhao/CodeDemo/TentNet-Tsingzao/data/'+data_name+'_test_'+str(nb_class)+'.mat'
    if pool=='average':
        temp = h5py.File(train_file)
        train_data = np.asarray(np.transpose(np.reshape(np.mean(np.transpose(temp['feature'],(2,1,0)),-1),(32,32,-1)),(2,1,0)),dtype='float32')
        train_label= temp['label']
        temp = h5py.File(test_file)
        test_data  = np.asarray(np.transpose(np.reshape(np.mean(np.transpose(temp['feature'],(2,1,0)),-1),(32,32,-1)),(2,1,0)),dtype='float32')
        test_label = temp['label']
    elif pool=='max':
        temp = h5py.File(train_file)
        train_data = np.asarray(np.transpose(np.reshape(np.max(np.transpose(temp['feature'],(2,1,0)),-1),(32,32,-1)),(2,1,0)),dtype='float32')
        train_label= temp['label']
        temp = h5py.File(test_file)
        test_data  = np.asarray(np.transpose(np.reshape(np.max(np.transpose(temp['feature'],(2,1,0)),-1),(32,32,-1)),(2,1,0)),dtype='float32')
        test_label = temp['label']
    elif pool=='none':
        temp = h5py.File(train_file)
        train_data = np.asarray(np.transpose(np.reshape(np.transpose(temp['feature'],(2,1,0)),(32,32,-1,16,1)),(2,1,0,3,4)),dtype='float32')
        train_label= temp['label']
        temp = h5py.File(test_file)
        test_data  = np.asarray(np.transpose(np.reshape(np.transpose(temp['feature'],(2,1,0)),(32,32,-1,16,1)),(2,1,0,3,4)),dtype='float32')
        test_label = temp['label']
    else:
        temp = h5py.File(train_file)
        train_data = np.asarray(np.transpose(np.transpose(temp['feature'],(2,1,0))[:,:,random.randint(0,15)].reshape(32,32,-1),(2,1,0)),dtype='float32')
        train_label= temp['label']
        temp = h5py.File(test_file)
        test_data  = np.asarray(np.transpose(np.transpose(temp['feature'],(2,1,0))[:,:,random.randint(0,15)].reshape(32,32,-1),(2,1,0)),dtype='float32')
        test_label = temp['label']
    return train_data, train_label, test_data, test_label

def data_processing(data,label,nb_class,pool='yes'):
    data /= 255
    data -= np.mean(data)
    if pool=='none':
        data = np.reshape(data,(-1,16,16,16,1))
    else:
        data = np.reshape(data,(-1,16,16,1))
    unique_label = np.unique(label)
    count = 0
    for i in unique_label:
        label[label==i]=count
        count += 1
    label = to_categorical(label,nb_class)
    return data,label

'''====================Train Deep Model===================================='''
size = 16
file_name = 'penn_'+str(size)+'x'+str(size)
nb_class  = 5
loss_term = 'categorical_crossentropy'

x_train, y_train, x_test, y_test = load_data(file_name, nb_class)
x_train, y_train = data_processing(x_train, y_train, nb_class)
x_test,  y_test  = data_processing(x_test, y_test, nb_class)

model_ln = lenet(nb_class)
model_ln.compile(optimizer='sgd', loss=loss_term, metrics=['accuracy'])
model_ln.fit(x_train[:,2:30,2:30,:], y_train, batch_size=4, epochs=50, validation_data=(x_test[:,2:30,2:30,:],y_test))

model_cn = cifarnet(nb_class)
model_cn.compile(optimizer='sgd', loss=loss_term, metrics=['accuracy'])
model_cn.fit(x_train, y_train, batch_size=4, epochs=50, validation_data=(x_test,y_test))

model_fc = fcnet(nb_class)
model_fc.compile(optimizer='sgd', loss=loss_term, metrics=['accuracy'])
model_fc.fit(np.reshape(x_train,(-1,1024)), y_train, batch_size=4, epochs=50, validation_data=(np.reshape(x_test,(-1,1024)), y_test))

#==============================================================================
# model_in = imagenet(nb_class)
# model_in.compile(optimizer='sgd', loss=loss_term, metrics=['accuracy'])
# model_in.fit(x_train, y_train, batch_size=4, epochs=10, validation_data=(x_test,y_test))
#==============================================================================

x_train, y_train, x_test, y_test = load_data(file_name, nb_class, pool='none')
x_train, y_train = data_processing(x_train, y_train, nb_class, pool='none')
x_test,  y_test  = data_processing(x_test, y_test, nb_class, pool='none')

model_3n = c3dnet(nb_class)
model_3n.compile(optimizer='sgd', loss=loss_term, metrics=['accuracy'])
model_3n.fit(x_train, y_train, batch_size=4, epochs=50, validation_data=(x_test,y_test))

model_tn = tentnet(nb_class)
model_tn.compile(optimizer='sgd', loss=loss_term, metrics=['accuracy'])
model_tn.fit(x_train, y_train, batch_size=4, epochs=50, validation_data=(x_test,y_test))

























