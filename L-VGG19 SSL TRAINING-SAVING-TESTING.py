# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 18:45:59 2020

@author: John
"""
print("[INFO] Importing Libraries")
import matplotlib as plt
import matplotlib.pyplot as plt
plt.style.use('ggplot')
# matplotlib inline
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import time   # time1 = time.time(); print('Time taken: {:.1f} seconds'.format(time.time() - time1))
import warnings
import keras
from keras.preprocessing.image import ImageDataGenerator
warnings.filterwarnings("ignore")
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.utils import to_categorical
from keras.preprocessing import image
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras import regularizers
from keras import optimizers
from keras.layers import LeakyReLU
from keras.layers import ELU
from keras.models import Model
from keras.layers import Input, Dense
from keras.layers import ZeroPadding2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from PIL import Image 
import numpy
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import time
from sklearn.metrics import classification_report, confusion_matrix
from keras_applications.resnet import ResNet50
from keras_applications.mobilenet import MobileNet
SEED = 50   # set random seed
print("[INFO] Libraries Imported")


#adam = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
#leakyrelu = keras.layers.LeakyReLU(alpha=0.3)
#elu = keras.layers.ELU(alpha=1.0)


input_img = Input(shape=(32, 32, 3)) 

from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.utils import plot_model

#%%
def make_model():
    base_model = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=(32, 32, 3), pooling=None, classes=2)
   
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])

    for layer in base_model.layers:
        layer.trainable = False
    for layer in base_model.layers[20:]:
        layer.trainable = True

    early2 = layer_dict['block2_pool'].output 

    early2 = BatchNormalization()(early2)
    early2 = Dropout(0.5)(early2)
    early2= GlobalAveragePooling2D()(early2)
        
    early3 = layer_dict['block3_pool'].output   

    early3 = BatchNormalization()(early3)
    early3 = Dropout(0.5)(early3)
    early3= GlobalAveragePooling2D()(early3)    
        
    early4 = layer_dict['block4_pool'].output   

    early4 = BatchNormalization()(early4)
    early4 = Dropout(0.5)(early4)
    early4= GlobalAveragePooling2D()(early4)     
        
        
        
        
    x1 = layer_dict['block5_conv3'].output 
    x1= GlobalAveragePooling2D()(x1)

    x = keras.layers.concatenate([x1, early4, early3], axis=-1)  
    
    
    x = Dense(2500, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(2, activation='softmax')(x)
    model = Model(input=base_model.input, output=x)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print("[INFO] Model Compiled!")
    return model    

   
    
def load_images(path):
    data = []
    labels = []
    SEED = 12
    # grab the image paths and randomly shuffle them
    imagePaths = sorted(list(paths.list_images(path)))
    random.seed(SEED)
    random.shuffle(imagePaths)
    
    
    # loop over the input images
    for imagePath in imagePaths:
        # load the image, resize the image to be 32x32 pixels (ignoring aspect ratio), 
        # flatten the 32x32x3=3072 pixel image into a list, and store the image in the data list
        image = cv2.imread(imagePath)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (32, 32))/255
        data.append(image)
     
        # extract the class label from the image path and update the labels list
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)
    
    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float")
    labels = np.array(labels)
    
    print("[INFO] Private data images loaded!")
    
    print("Reshaping data!")
    
    #data = data.reshape(data.shape[0], 32, 32, 1)
    
    print("Data Reshaped to feed into models channels last")
    
    print("Labels formatting")
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels) 
    labels = keras.utils.to_categorical(labels, num_classes=2, dtype='float32')  
    
    return data, labels  
    





def pick_reliable(prediction_num, fake_data, threshold):#pick the most reliable predictions and return them with the images. returns new dataU
    data=[]
    labels=[]
    dataU = []
    
    
    
    for i in range(len(fake_data)):
        if prediction_num[i,0]>threshold:
            f = fake_data[i,:,:,:]
            data.append(f)
            label = 'benign'
            labels.append(label)
        elif prediction_num[i,1]>threshold:
            f = fake_data[i,:,:,:]
            data.append(f)
            label = 'malignant'
            labels.append(label) 
        else: 
            f = fake_data[i,:,:,:]
            dataU.append(f)
            
    data = np.array(data, dtype="float")
    dataU = np.array(dataU, dtype="float") 
    labels = np.array(labels)
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels) 
    labels = keras.utils.to_categorical(labels, num_classes=2, dtype='float32')
    #labels = np.argmax(labels, axis=-1)
    return data, labels, dataU
            
       



         
def merge_datas(data, labels, datanext, labelsnext):#merge the previous training data and labels with the new
    
    mergeX = np.concatenate([data, datanext])
    mergeY = np.concatenate([labels, labelsnext])
    return mergeX, mergeY
  



  
    
def train_self (dataL, dataU, labelsL, epochs):#train the model on any labelled data. returns the accuracy and the predicitions on unlabelled data
    
    
    data = dataL
    labels = labelsL
    
    n_split=2 #10fold cross validation
    scores = [] #here every fold accuracy will be kept
    predictions_all = np.empty(0) # here, every fold predictions will be kept
    predictions_all_num = np.empty([0,2]) # here, every fold predictions will be kept
    test_labels = np.empty(0) #here, every fold labels are kept
    name2 = 5000 #name initiator for the incorrectly classified insatnces
    conf_final = np.array([[0,0],[0,0]]) #initialization of the overall confusion matrix
    for train_index,test_index in KFold(n_split).split(data):
        trainX,testX=data[train_index],data[test_index]
        trainY,testY=labels[train_index],labels[test_index]
        model3 = make_model()
    
        aug = ImageDataGenerator(rotation_range=45, horizontal_flip=True, vertical_flip=True, fill_mode = 'nearest')
        aug.fit(trainX)
        model3.fit_generator(aug.flow(trainX, trainY,batch_size=64), epochs=epochs, steps_per_epoch=len(trainX)//64)
        score = model3.evaluate(testX,testY)
        score = score[1] #keep the accuracy score, not the loss
        scores.append(score) #put the fold score to list
        testY2 = np.argmax(testY, axis=-1) #make the labels 1column array
        print('Model evaluation ',model3.evaluate(testX,testY))
        predict = model3.predict(testX) #for def models functional api
        predict_num = predict
        predict = predict.argmax(axis=-1) #for def models functional api
        conf = confusion_matrix(testY2, predict) #get the fold conf matrix
        conf_final = conf + conf_final #sum it with the previous conf matrix
        name2 = name2 + 1
        predictions_all = np.concatenate([predictions_all, predict]) #merge the two np arrays of predicitons
        predictions_all_num = np.concatenate([predictions_all_num, predict_num])
        test_labels = np.concatenate ([test_labels, testY2]) #merge the two np arrays of labels
    scores = np.asarray(scores)
    final_score = np.mean(scores)  
    model3.save('self_vgg19fe.h5')

    predict_fake = model3.predict(dataU)

    return final_score, predict_fake, conf_final    


# def predict (data):
#     vgg = load_model('self_vgg19fe.h5')
#     pred = vgg.predict(data)
#     return pred
    
def train_self2 (dataL1, dataU, labelsL1, dataL, labelsL, best_pred, pred_labels, epochs):#train the model on any labelled data. returns the accuracy and the predicitions on unlabelled data
    
    data = dataL1
    labels = labelsL1
    
    

    
    n_split=2 #10fold cross validation
    scores = [] #here every fold accuracy will be kept
    predictions_all = np.empty(0) # here, every fold predictions will be kept
    predictions_all_num = np.empty([0,2]) # here, every fold predictions will be kept
    test_labels = np.empty(0) #here, every fold labels are kept
    name2 = 5000 #name initiator for the incorrectly classified insatnces
    conf_final = np.array([[0,0],[0,0]]) #initialization of the overall confusion matrix
    for train_index,test_index in KFold(n_split).split(data):
        trainX,testX=data[train_index],data[test_index]
        trainY,testY=labels[train_index],labels[test_index]
        
        #data, labels = merge_datas(dataL,labelsL, dataLn, labelsLn)
        
        trainX = np.concatenate([trainX, dataL, best_pred])
        trainY = np.concatenate([trainY, labelsL, pred_labels ])
        
        
        
        model3 = make_model()
    
        aug = ImageDataGenerator(rotation_range=45, horizontal_flip=True, vertical_flip=True, fill_mode = 'nearest')
        aug.fit(trainX)
        model3.fit_generator(aug.flow(trainX, trainY,batch_size=64), epochs=epochs, steps_per_epoch=len(trainX)//64)
        score = model3.evaluate(testX,testY)
        score = score[1] #keep the accuracy score, not the loss
        scores.append(score) #put the fold score to list
        testY2 = np.argmax(testY, axis=-1) #make the labels 1column array
        print('Model evaluation ',model3.evaluate(testX,testY))
        predict = model3.predict(testX) #for def models functional api
        predict_num = predict
        predict = predict.argmax(axis=-1) #for def models functional api
        conf = confusion_matrix(testY2, predict) #get the fold conf matrix
        conf_final = conf + conf_final #sum it with the previous conf matrix
        name2 = name2 + 1
        predictions_all = np.concatenate([predictions_all, predict]) #merge the two np arrays of predicitons
        predictions_all_num = np.concatenate([predictions_all_num, predict_num])
        test_labels = np.concatenate ([test_labels, testY2]) #merge the two np arrays of labels
    scores = np.asarray(scores)
    final_score = np.mean(scores)  
    model3.save('self_vgg19fe.h5')

    predict_fake = model3.predict(dataU)

    return final_score, predict_fake, conf_final 



#%% train self

#load labelled data
path = 'C:\\Users\\User\\gan_classification_many datasets\\PET'
dataL1,labelsL1 = load_images(path)

#load unlabelled data initial
path2 = 'C:\\Users\\User\\Gan_tritraining Lvgg19\\GAN labeled images (separate training for each class)'
dataU1,labelsU1 = load_images(path2)

# how many attempts to concatenate new instances
iterations = 2

#%%
#self training
epochs = 1
threshold = 0.995

dataL = dataL1
labelsL = labelsL1
dataU = dataU1

#%%
for i in range (iterations):
    
    print('[INFO] Iteration Number %2d'%i)
    if i==0:
        final_score, predict_num, conf_final = train_self (dataL1, dataU, labelsL1, epochs)
        print('[INFO] Accuracy on PET: %2.4f' %final_score) 
    
    
       
    best_pred, pred_labels, dataU = pick_reliable(predict_num,dataU,threshold)

    print('[INFO] Picked %5d reliable predictions' %len(best_pred))
    #dataL, labelsL = merge_datas(dataL,labelsL, best_pred, pred_labels)
    op = len(dataL)+len(best_pred)
    print('[INFO] Labelled Training Data increased to %8d' %op)
    print('[INFO] Training with the expanded data...')          
    final_score, predict_num, conf_final = train_self2 (dataL1, dataU, labelsL1, dataL, labelsL, best_pred, pred_labels, epochs)
   
    print('[INFO] New Accuracy: %2.4f' %final_score)
    dataL, labelsL = merge_datas(dataL,labelsL, best_pred, pred_labels)
    print('[INFO] Merging the newly labelled data to a different labelled set...')
    print('[INFO] The initial datasize is: %4d, the new labelled data size is: %5d , remaining unlabelled instances: %5d' %(len(dataL1), len(dataL), len(dataU)))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
   
    
    