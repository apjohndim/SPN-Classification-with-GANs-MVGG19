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
    early2 = layer_dict['block2_pool'].output 
    #early2 = Conv2D(64, (3, 3), padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.001))(early2)
    early2 = BatchNormalization()(early2)
    early2 = Dropout(0.5)(early2)
    early2= GlobalAveragePooling2D()(early2)  
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

def make_mobilenet():
    base_model = keras.applications.mobilenet.MobileNet(input_shape=(32, 32, 3), alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=False, weights='imagenet', input_tensor=None, pooling=None, classes=2)
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
                   
    for layer in base_model.layers:
        layer.trainable = False

    x = layer_dict['conv_pw_6'].output
    x= GlobalAveragePooling2D()(x)

    #x = BatchNormalization()(x)
    #x = Flatten()(x)
    x = Dense(2500, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(2, activation='softmax')(x)
    model = Model(input=base_model.input, output=x)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
   # model.summary()

    return model


from keras.applications.densenet import DenseNet121

def make_dense():
    base_model = keras.applications.densenet.DenseNet121(include_top=False, weights='imagenet', input_tensor=None, input_shape=(32, 32, 3), pooling=None, classes=2)
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    for layer in base_model.layers:
        layer.trainable = False
    x = layer_dict['conv4_block24_concat'].output
    x= GlobalAveragePooling2D()(x)

    x = Dense(2500, activation='relu')(x)
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


    
def train_all(modellek,Targetdata,TargetLabels, merge, data2, labels2, aug):
    
    data = Targetdata
    labels = TargetLabels
    
    n_split=10 #10fold cross validation
    
    
    def tr(model1,merge):
        
        scores = [] #here every fold accuracy will be kept
        predictions_all = np.empty(0) # here, every fold predictions will be kept
        predictions_all_num = np.empty([0,2]) # here, every fold predictions will be kept
        test_labels = np.empty(0) #here, every fold labels are kept
        name2 = 5000 #name initiator for the incorrectly classified insatnces
        conf_final = np.array([[0,0],[0,0]]) #initialization of the overall confusion matrix
        
        
        for train_index,test_index in KFold(n_split).split(data):
            trainX,testX=data[train_index],data[test_index]
            trainY,testY=labels[train_index],labels[test_index]   
            
            
            if model1 == 'lvgg':
               model = make_lvgg()
            if model1 == 'vgg16fe':
               model = make_vgg16fe()
            if model1 == 'vgg19fe':
               model = make_vgg19fe()
            if model1 == 'zhao':
               model = make_zhao()
            
            #if merge == 1:
            trainX = np.concatenate([trainX, data2])
            trainY = np.concatenate([trainY, labels2])

            #aug = ImageDataGenerator(rotation_range=45, horizontal_flip=True, vertical_flip=True, fill_mode = 'nearest')
            #aug.fit(trainX)
            #model.fit_generator(aug.flow(trainX, trainY,batch_size=64), epochs=50, steps_per_epoch=len(trainX)//64)
            print ('Model Created')
            model.fit(trainX, trainY,epochs=4, batch_size=64)
            
            
            time.sleep(2) 
            score = model.evaluate(testX,testY)
            time.sleep(2) 
            score = score[1] #keep the accuracy score, not the loss
            scores.append(score) #put the fold score to list
            testY2 = np.argmax(testY, axis=-1) #make the labels 1column array
            print('Model evaluation ',model.evaluate(testX,testY))
            predict = model.predict(testX) #for def models functional api
            predict_num = predict
            predict = predict.argmax(axis=-1) #for def models functional api
            conf = confusion_matrix(testY2, predict) #get the fold conf matrix
            conf_final = conf + conf_final #sum it with the previous conf matrix
            name2 = name2 + 1
            predictions_all = np.concatenate([predictions_all, predict]) #merge the two np arrays of predicitons
            predictions_all_num = np.concatenate([predictions_all_num, predict_num])
            test_labels = np.concatenate ([test_labels, testY2]) #merge the two np arrays of labels
            print ('Model: '+ str (model1) + ' - Fold number: ' + str (name2 - 5000))
            time.sleep(1) 
        scores = np.asarray(scores)
        final_score = np.mean(scores)  
        #model3.save('self_vgg19fe.h5')
        return final_score, conf_final 
    
    
    
    if modellek == 'lvgg':
       f1,c1 = tr(modellek,merge)
       print ('Finished Evaluating Model: ' + str (modellek))
    if modellek == 'vgg16fe':
       f1,c1 = tr(modellek,merge)
       print ('Finished Evaluating Model: ' + str (modellek))
    if modellek == 'vgg19fe':
       f1,c1 = tr(modellek,merge)
       print ('Finished Evaluating Model: ' + str (modellek))
    if modellek == 'zhao':
       f1,c1 = tr(modellek,merge)
       print ('Finished Evaluating Model: ' + str (modellek))
    

    return  f1,c1


def testallou (model, data,labels,testX, testY):
    
    if model == 'lvgg':
       model = make_lvgg()
    if model == 'vgg16fe':
       model = make_vgg16fe()
    if model == 'vgg19fe':
       model = make_vgg19fe()
    if model == 'zhao':
       model = make_zhao()
    
    model.fit(data, labels,epochs=6, batch_size=64)
    score = model.evaluate(testX,testY)
    testY2 = np.argmax(testY, axis=-1) #make the labels 1column array
    print('Model evaluation ',score)
    predict = model.predict(testX)
    predict = predict.argmax(axis=-1) #for def models functional api
    conf = confusion_matrix(testY2, predict) #get the fold conf matrix
    
    return score[1], conf


#%%
    
path1 = 'C:\\Users\\User\\gan_classification_many datasets\\OLD DATASETS\\LIDC_NEW'
path2 = 'C:\\Users\\User\\Gan_tritraining Lvgg19\\GAN labeled images (LVGG19 labelled without self_tr)'

path3 = 'C:\\Users\\User\\gan_classification_many datasets\\PET'
aug = 0
merge = 0

data,labels = load_images(path1)
data2,labels2 = load_images(path2)
data3,labels3 = load_images(path3)

data2 = np.concatenate([data2, data3])
labels2 = np.concatenate([labels2, labels3])



#%% TRAIN AND TEST ON THE SAME DATA (PLUS CONCATENATION OF NEW)


modellek = 'lvgg'
f1,c1 = train_all(modellek,data,labels,0,data2,labels2,1) #data prwta data tha pathoun kfold. ta deutera tha ginoun concat
    

#%% TRAIN ON A, TEST ON B

testX = data
testY = labels


modelo = 'lvgg'
score,conf = testallou (modelo,data2,labels2,testX,testY) #ta prwta data einai ekei pou tha ekpaideytei, ta deutera einai ekei pou tha metrithei