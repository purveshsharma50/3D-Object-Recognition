#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 23:06:57 2020

@author: purveshsharma
"""


#------------------------------------------------------------------------------------------------------------------
#Importing different libraries

# import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Concatenate
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras import Input
from keras import optimizers
import random
from tensorflow import keras
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
# import seaborn as sn
import itertools
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator


 
#--------------------------------------------------------------------------------------------------------------------
# Importing Dataset 


# For LEAP Server
# DATADIR_RGB= "/gpfs/home/p_s218/Training_Dataset/RGB/"     # RGB Dataset for LEAP cluster
# DATADIR_D = "/gpfs/home/p_s218/Training_Dataset/Depth/"    # Depth Dataset for LEAP cluster



# To classify the folders in the path
CATEGORIES = ["bowl","soda","glue","Dove"]

#--------------------------------------------------------------------------------------------------------------------
#Pre-Processing for RGB image

IMG_SIZE = 150       # Image resolution

training_data = []

# This process is to combine all the images in each class to feed it to the network by creating a new variable training data  
def create_training_data():
    for category in CATEGORIES:  

        path = os.path.join(DATADIR_RGB,category) 
        class_num = CATEGORIES.index(category)  
        print (class_num)
        for img in tqdm(os.listdir(path)): 
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.COLOR_BGR2RGB)        # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))                  # resize to normalize data size
                alpha = random.random()                                                  # Change in contrast
                beta = random.randint(0, 100)                                            # Change in brightness
                new_array2 = cv2.convertScaleAbs(new_array, alpha = alpha, beta = beta)
                training_data.append([new_array2, class_num])                            # add this to our training_data
            except Exception as e:                                                       # in the interest in keeping the output clean...
                pass
            
create_training_data()


X_train = []
y_train = []

for features,label in training_data:
    X_train.append(features)            # Features are the array value for each image
    y_train.append(label)

#print(X_train[0].reshape(-1, IMG_SIZE, IMG_SIZE, 3))

X_train = np.array(X_train).reshape(-1, IMG_SIZE, IMG_SIZE, 3)                            # Reshaping the array
y_train = np.array(y_train)


X_train = X_train/255.0  

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state= 1)


#------------------------------------------------------------------------------------------------------------------
#Pre- Processing for Depth Image

IMG_SIZE_D = 150
training_data_D = []

#for training 
def create_training_data_D():
    for category in CATEGORIES: 

        path = os.path.join(DATADIR_D,category)      # create path  
        class_num = CATEGORIES.index(category)  
        print (class_num)
        for img in tqdm(os.listdir(path)):  
            try:
                img_array_D = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)      # convert to array
                new_array_D = cv2.resize(img_array_D, (IMG_SIZE_D, IMG_SIZE_D))             # resize to normalize data size
                training_data_D.append([new_array_D, class_num])                            # add this to our training_data
            except Exception as e:                                                          # in the interest in keeping the output clean...
                pass
            
create_training_data_D()


X_train_D = []

for features,label in training_data_D:
    X_train_D.append(features)



X_train_D = np.array(X_train_D).reshape(-1, IMG_SIZE_D, IMG_SIZE_D, 1)

X_train_D = X_train_D/255.0  
X_train_D, X_test_D = train_test_split(X_train_D, test_size=0.3, random_state= 1)

#--------------------------------------------------------------------------------------------------------------------
# Main Architecture Design
def modelRGBD():
    inp_RGB = Input(shape =(IMG_SIZE , IMG_SIZE, 3))
    
    conv_layer1 = Conv2D(8, (7,7), strides=(1, 1), padding='valid', activation='relu')(inp_RGB)
    conv_layer2 = Conv2D(8, (7,7), strides=(1, 1), padding='valid', activation='relu')(conv_layer1)
    #BN_1 = BatchNormalization()(conv_layer2)
    pool_layer1 = MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='valid')(conv_layer2)
    dropout1 = Dropout(0.5)(pool_layer1)
    
    
    conv_layer3 =Conv2D(8, (3,3), strides=(1, 1), padding='valid', activation='relu')(dropout1)
    conv_layer4 = Conv2D(16, (3,3), strides=(1, 1), padding='valid', activation='relu')(conv_layer3)
    #BN_2 = BatchNormalization()(conv_layer4)
    pool_layer2 = MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='valid')(conv_layer4)
    dropout2 = Dropout(0.2)(pool_layer2)
    
    
    
    conv_layer5 = Conv2D(32, (3,3), strides=(1, 1), padding='valid', activation='relu')(dropout2)
    #BN_3 = BatchNormalization()(conv_layer5)
    pool_layer3 = MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='valid')(conv_layer5)
    dropout3 = Dropout(0.2)(pool_layer3)
    
    
    flatten_layer = Flatten()(dropout3)
    hidden1 = Dense(512, activation = 'relu')(flatten_layer)
    #BN_4 = BatchNormalization()(hidden1)
    dropout4 = Dropout(0.2)( hidden1)
    


    inp_D = Input(shape =(IMG_SIZE , IMG_SIZE, 1))
    
    conv_layer1_D = Conv2D(8, (7,7), strides=(1, 1), padding='valid', activation='relu')(inp_D)
    conv_layer2_D = Conv2D(8, (7,7), strides=(1, 1), padding='valid', activation='relu')(conv_layer1_D)
    #BN_1 = BatchNormalization()(conv_layer2)
    pool_layer1_D = MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='valid')(conv_layer2_D)
    dropout1_D = Dropout(0.5)(pool_layer1_D)
    
    
    conv_layer3_D =Conv2D(16, (3,3), strides=(1, 1), padding='valid', activation='relu')(dropout1_D)
    conv_layer4_D = Conv2D(16, (3,3), strides=(1, 1), padding='valid', activation='relu')(conv_layer3_D)
    #BN_2 = BatchNormalization()(conv_layer4)
    pool_layer2_D = MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='valid')(conv_layer4_D)
    dropout2_D = Dropout(0.2)(pool_layer2_D)
    
    
    
    conv_layer5_D = Conv2D(32, (3,3), strides=(1, 1), padding='valid', activation='relu')(dropout2_D)
    #BN_3 = BatchNormalization()(conv_layer5)
    pool_layer3_D = MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='valid')(conv_layer5_D)
    dropout3_D = Dropout(0.2)(pool_layer3_D)
    
    
    flatten_layer_D = Flatten()(dropout3_D)
    hidden1_D = Dense(512, activation = 'relu')(flatten_layer_D)
    #BN_4 = BatchNormalization()(hidden1)
    dropout4_D = Dropout(0.2)( hidden1_D)

    hidden_merge = Concatenate(axis = -1)([dropout4 , dropout4_D])             # Combining both streams

    hidden = Dense(1024,activation = 'relu')(hidden_merge)
    #BN = BatchNormalization()(hidden)
    dropout = Dropout(0.5)(hidden)
    #hidden = Dense(1024,activation = 'relu')(dropout)
    out = Dense(4,activation='softmax')(dropout)

    model1 = Model([inp_RGB, inp_D],out)
    
    model1.summary()
    plot_model(model1, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9)
    model1.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['acc'])
    
    return model1
 

model = modelRGBD()

history=model.fit([X_train,X_train_D], y_train, batch_size=128, epochs=3, validation_data=([X_test, X_test_D],y_test))

result= model.evaluate([X_test,X_test_D], y_test, verbose = 0)
print(result)



#---------------------------------------------------------------------------------------------------------------------
# To Plot the model

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

y_pred= model.predict([X_test,X_test_D])
y_pred = np.argmax(y_pred, axis=1)

print('Confusion Matrix')
target_names = ["bowl","soda","glue","Dove"]


cm = confusion_matrix(y_test, y_pred)


def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('Confusion Matrics.png')
    plt.close()
    
cm_plot_labels = ['bowl','soda','glue','Dove']

plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')


# print(confusion_matrix(y_test, y_pred))
print('Classification Report')
print(classification_report(y_test, y_pred, target_names=target_names))


plt.plot(epochs, acc, 'g', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Testing acc')
plt.title('Training and Testing accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()    
# plt.figure()
plt.savefig('Training and Testing accuracy.png')
plt.close()

plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'k', label='Testing loss')
plt.title('Training and Testing loss')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
# plt.show()
plt.savefig('Training and Testing loss.png')
plt.close()
