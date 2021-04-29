
print("[INFO] Importing Libraries")
import matplotlib as plt
plt.style.use('ggplot')
# matplotlib inline
from sklearn.preprocessing import LabelBinarizer
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
import tensorflow as tf
SEED = 50   # set random seed
print("[INFO] Libraries Imported")


#adam = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
#leakyrelu = keras.layers.LeakyReLU(alpha=0.3)
#elu = keras.layers.ELU(alpha=1.0)


from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.utils import plot_model

def make_zhao():
    base_model = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=(32, 32, 3), pooling=None, classes=2)
    #base_model = keras.applications.resnet.ResNet50(include_top=False, weights='imagenet', input_shape=(32, 32, 3), pooling='max')
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    #base_model.summary()
    #plot_model(base_model, to_file='basevgg16.png')
    
    
    #base_model.summary()
    for layer in base_model.layers[:]:
        layer.trainable = False
        
    msb1 = layer_dict['block1_pool'].output
    msb1 = Conv2D(128, (3, 3), padding='valid', activation='relu', strides=4)(msb1)
    msb1 = Conv2D(128, (1, 1), padding='valid', activation='relu', strides=2)(msb1)
    msb1 = Conv2D(512, (1, 1), padding='valid', activation='relu', strides=2)(msb1)
    
    msb2 = layer_dict['block2_pool'].output
    msb2 = Conv2D(128, (3, 3), padding='valid', activation='relu', strides=4)(msb2)
    msb2 = Conv2D(128, (1, 1), padding='valid', activation='relu', strides=2)(msb2)
    msb2 = Conv2D(512, (1, 1), padding='valid', activation='relu', strides=2)(msb2)
    
    msb3 = layer_dict['block3_pool'].output
    msb3 = Conv2D(128, (3, 3), padding='valid', activation='relu', strides=2)(msb3)
    msb3 = Conv2D(128, (1, 1), padding='valid', activation='relu', strides=2)(msb3)
    msb3 = Conv2D(512, (1, 1), padding='valid', activation='relu', strides=1)(msb3)
    
    msb4 = layer_dict['block4_pool'].output
    msb4 = Conv2D(128, (2, 2), padding='valid', activation='relu', strides=1)(msb4)
    msb4 = Conv2D(128, (1, 1), padding='valid', activation='relu', strides=1)(msb4)
    msb4 = Conv2D(512, (1, 1), padding='valid', activation='relu', strides=1)(msb4)
    
    x = layer_dict['block5_pool'].output
    
    x = keras.layers.concatenate([msb1, msb2, msb3, msb4, x], axis=3)
    
    x= GlobalAveragePooling2D()(x)
    #x = Flatten()(x)
    #x = Dense(256, activation='relu')(x)
    
    x = Dense(4096, activation='relu')(x)
    x = BatchNormalization()(x)
    #x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = BatchNormalization()(x)
    
    x = Dense(2, activation='softmax')(x)
    model = Model(input=base_model.input, output=x)
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #plot_model(model, to_file='vggmod1.png')
    return model
 

def make_lvgg():
    base_model = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=(32, 32, 3), pooling=None, classes=2)
    #base_model = keras.applications.resnet.ResNet50(include_top=False, weights='imagenet', input_shape=(32, 32, 3), pooling='max')
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    #base_model.summary()
    for layer in base_model.layers:
        layer.trainable = False
    for layer in base_model.layers[20:]:
        layer.trainable = True
    #base_model.summary()
    
    # early1 = layer_dict['block1_pool'].output
    # #early1 = Conv2D(32, (3, 3), padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.001))(early1)
    # early1 = BatchNormalization()(early1)
    # early1 = Dropout(0.5)(early1)
    # early1= GlobalAveragePooling2D()(early1)
    # #early1 = Flatten()(early1)
    #early2 = layer_dict['block2_pool'].output 
    #early2 = Conv2D(64, (3, 3), padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.001))(early2)
    #early2 = BatchNormalization()(early2)
    #early2 = Dropout(0.5)(early2)
    #early2= GlobalAveragePooling2D()(early2)  
        
    early3 = layer_dict['block3_pool'].output   
    #early3 = Conv2D(128, (3, 3), padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.001))(early3)
    early3 = BatchNormalization()(early3)
    early3 = Dropout(0.5)(early3)
    early3= GlobalAveragePooling2D()(early3)
    
    early4 = layer_dict['block4_pool'].output   
    #early4 = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001))(early4)
    early4 = BatchNormalization()(early4)
    early4 = Dropout(0.5)(early4)
    early4= GlobalAveragePooling2D()(early4)
     
    x1 = layer_dict['block5_conv3'].output 
    x1= GlobalAveragePooling2D()(x1)
    #x1 = Flatten()(x1)
    x = keras.layers.concatenate([x1, early4, early3], axis=-1)
    #x = GlobalAveragePooling2D()(x) 

    #x = Flatten()(x)
    #x = Dense(256, activation='relu')(x)
    x = Dense(2500, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(2, activation='softmax')(x)
    model = Model(input=base_model.input, output=x)
    #for layer in model.layers[:17]:
        #layer.trainable = True
    
    # for layer in model.layers[17:]:
    #     layer.trainable = True  
    #model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #plot_model(model, to_file='vggmod19.png')
    return model

def make_vgg16fe():
    base_model = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=(32, 32, 3), pooling=None, classes=2)
    #base_model = keras.applications.resnet.ResNet50(include_top=False, weights='imagenet', input_shape=(32, 32, 3), pooling='max')
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    #base_model.summary()
    #plot_model(base_model, to_file='basevgg16.png')
    
    
    #base_model.summary()
    
    x = layer_dict['block5_conv3'].output
    x= GlobalAveragePooling2D()(x)
    #x = Flatten()(x)
    #x = Dense(256, activation='relu')(x)
    
    x = Dense(2500, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(2, activation='softmax')(x)
    model = Model(input=base_model.input, output=x)
    
    for layer in model.layers[:17]:
        layer.trainable = False
     
    for layer in model.layers[17:]:
        layer.trainable = True  
    #model.summary()
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #plot_model(model, to_file='vggmod1.png')
    return model

def make_vgg19fe():
    base_model = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=(32, 32, 3), pooling=None, classes=2)
    #base_model = keras.applications.resnet.ResNet50(include_top=False, weights='imagenet', input_shape=(32, 32, 3), pooling='max')
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    for layer in base_model.layers:
        layer.trainable = False
    for layer in base_model.layers[20:]:
        layer.trainable = True
    
    x1 = layer_dict['block5_conv3'].output 
    x1= GlobalAveragePooling2D()(x1)

    x = Dense(2500, activation='relu')(x1)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(2, activation='softmax')(x)
    model = Model(input=base_model.input, output=x)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #plot_model(model, to_file='vggmod19.png')
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
        image = cv2.resize(image, (32, 32))/215
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
        if prediction_num[i,0]>threshold and prediction_num[i,1]<1 - threshold :
            f = fake_data[i,:,:,:]
            data.append(f)
            label = 'benign'
            labels.append(label)
        elif prediction_num[i,1]>threshold and prediction_num[i,0] < 1- threshold:
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
  



  
    
def train_self (dataL, dataU, labelsL, datahidden, labelshidden, epochs):#train the model on any labelled data. returns the accuracy and the predicitions on unlabelled data
    
    
    data = dataL
    labels = labelsL
    scores = [] #here every fold accuracy will be kept

    model3 = make_lvgg()
    
    
    model3.fit(data, labels,epochs=15, batch_size=64)
    time.sleep(2) 
    

    
    
    
    #aug = ImageDataGenerator(rotation_range=45, horizontal_flip=True, vertical_flip=True, fill_mode = 'nearest')
    #aug.fit(trainX)
    #model3.fit_generator(aug.flow(trainX, trainY,batch_size=64), epochs=epochs, steps_per_epoch=len(trainX)//64)
    
    score = model3.evaluate(datahidden,labelshidden)
    score = score[1] #keep the accuracy score, not the loss
    scores.append(score) #put the fold score to list
    testY2 = np.argmax(labelshidden, axis=-1) #make the labels 1column array
    print('Model evaluation ',score)
    predict = model3.predict(datahidden) #for def models functional api
    predict = predict.argmax(axis=-1) #for def models functional api
    conf = confusion_matrix(testY2, predict) #get the fold conf matrix
    

    predict_fake = model3.predict(dataU)

    return score, predict_fake, conf   


# def predict (data):
#     vgg = load_model('self_vgg19fe.h5')
#     pred = vgg.predict(data)
#     return pred
    
def train_self2 (dataL1, dataU, labelsL1, dataL, labelsL, best_pred, pred_labels, epochs, datahidden, labelshidden):#train the model on any labelled data. returns the accuracy and the predicitions on unlabelled data
    
    data = dataL1
    labels = labelsL1
    
    
    #data_ext,labels_ext = load_images('')
    

    scores = [] #here every fold accuracy will be kept

    
    #data, labels = merge_datas(dataL,labelsL, dataLn, labelsLn)
    
    trainX = np.concatenate([data, dataL, best_pred])
    trainY = np.concatenate([labels, labelsL, pred_labels])
    
    #trainX = np.concatenate([trainX, dataL, best_pred,datahidden])
    #trainY = np.concatenate([trainY, labelsL, pred_labels,labelshidden])

    model3 = make_lvgg()

    #aug = ImageDataGenerator(rotation_range=45, horizontal_flip=True, vertical_flip=True, fill_mode = 'nearest')
    #aug.fit(trainX)
    #model3.fit_generator(aug.flow(trainX, trainY,batch_size=64), epochs=5, steps_per_epoch=len(trainX)//32)
    model3.fit(trainX, trainY,epochs=epochs, batch_size=64)
    time.sleep(2) 
    
    testX = datahidden
    testY = labelshidden
    score = model3.evaluate(testX,testY)
    score = score[1] #keep the accuracy score, not the loss
    scores.append(score) #put the fold score to list
    testY2 = np.argmax(testY, axis=-1) #make the labels 1column array
    print('Model evaluation ',score)
    predict = model3.predict(testX) #for def models functional api

    predict = predict.argmax(axis=-1) #for def models functional api
    conf = confusion_matrix(testY2, predict) #get the fold conf matrix

    
    time.sleep(2) 
    predict_fake = model3.predict(dataU)
    time.sleep(5) 
    return score, predict_fake, conf



#%% train self

#load labelled data
path = 'C:\\Users\\User\\gan_classification_many datasets\\OLD DATASETS\\LIDC_NEW'
dataL1,labelsL1 = load_images(path)

#load unlabelled data initial
path2 = 'C:\\Users\\User\\Gan_tritraining Lvgg19\\GAN_EXPV2.1'
dataU1,labelsU1 = load_images(path2)


path3 = 'C:\\Users\\User\\gan_classification_many datasets\\PET'
datahidden,labelshidden = load_images(path3)
# how many attempts to concatenate new instances
iterations = 10

#self training
epochs = 4
threshold = 0.9991

dataL = dataL1
labelsL = labelsL1
dataU = dataU1

#%%

for i in range (iterations):
    
    print('[INFO] Iteration Number %2d'%i)
    if i==0:
        final_score, predict_num, conf_final = train_self (dataL1, dataU, labelsL1, datahidden,labelshidden, epochs)
        print('[INFO] Accuracy on PET: %2.4f' %final_score) 
        old_score = final_score
    
       
    best_pred, pred_labels, dataU = pick_reliable(predict_num,dataU,threshold)

    print('[INFO] Picked %5d reliable predictions' %len(best_pred))
    #dataL, labelsL = merge_datas(dataL,labelsL, best_pred, pred_labels)
    op = len(dataL)+len(best_pred)
    print('[INFO] Labelled Training Data increased to %8d' %op)
    print('[INFO] Training with the expanded data...')

    #with tf.device('/cpu:0'):        
    final_score, predict_num, conf_final = train_self2 (dataL1, dataU, labelsL1, dataL, labelsL, best_pred, pred_labels, epochs, datahidden, labelshidden)
   
    print('[INFO] New Accuracy: %2.4f' %final_score)
    
    if old_score < final_score:
        dataL, labelsL = merge_datas(dataL,labelsL, best_pred, pred_labels)
        old_score = final_score
        print('[INFO] Merging the newly labelled data to a different labelled set...')
        print('[INFO] The initial datasize is: %4d, the new labelled data size is: %5d , remaining unlabelled instances: %5d' %(len(dataL1), len(dataL), len(dataU)))
    else: print ('[INFO] Accuracy dropped. The new picks are removed. The new labelled data size is: ' + str (len(dataL)) + '. The old labelled data size is: ' + str(len(dataL1)))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
   
    
    