"Cerebellum Segmentation using M-net"

import os
import cv2
import glob
import h5py as hp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
import pdb
import tensorflow as tf

import keras as keras
from keras.models import Model
from keras.layers import Input, Reshape, ZeroPadding3D	, Activation, Conv2D, Conv3D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.optimizers import SGD, Adam
from keras import losses
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import nibabel as nib

# image_data_format = channels_last
K.set_image_data_format('channels_last')
print keras.backend.image_data_format()

##############################################################################
##############################################################################

def weighted_pixelwise_crossentropy(class_weights):
    
    def loss(y_true, y_pred):
        epsilon =  tf.convert_to_tensor(keras.backend.epsilon(), y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        return - tf.reduce_sum(tf.multiply(y_true * tf.log(y_pred), [1,1000]))
        # return - tf.reduce_sum((y_true * tf.log(y_pred)))
    return loss

'''
Defining some helper functions
'''
def my_to_categorical(y, nb_classes=None):
    print "1"
    Y = np.zeros([y.shape[0],y.shape[1],y.shape[2],y.max()+1],dtype='uint8')
    for i in range(0, y.shape[0]):
        print i
        for j in range(0, y.shape[1]):
            for k in range(0, y.shape[2]):
                if(y[i][j][k] == 1):
                    Y[i][j][k][1]=1
                    # print "Cerebellum"
                elif y[i][j][k] == 0:
                    Y[i][j][k][0]=1
                else : 
                    print "ERROR"

    print "2"
    # Y[np.nonzero(y)[0],np.nonzero(y)[1],np.nonzero(y)[2],y[np.nonzero(y)[0],np.nonzero(y)[1],np.nonzero(y)[2]]] = 1
    print "3"
    # Y[np.where(y == 0)[0],np.where(y == 0)[1],np.where(y == 0)[2],y[np.where(y == 0)[0],np.where(y == 0)[1],np.where(y == 0)[2]]] = 1  
    print "4"
    return Y

def accuracy(Y_pre, Y_true):
    accu = ((np.sum(Y_true[np.nonzero(Y_true)]==Y_pre[np.nonzero(Y_true)],dtype='float32'))/(np.count_nonzero(Y_true)))
    zero_accu = ((np.sum(Y_true[np.where(Y_true == 0)]==Y_pre[np.where(Y_true == 0)],dtype='float32'))/(np.size(Y_true)-np.count_nonzero(Y_true)))
    return [accu,zero_accu]

def dice_coef(y_true, y_pred, smooth=1):
    # y_true = K.flatten(y_true)
    # y_pred = K.flatten(y_pred)
    intersection = np.dot(y_true, np.transpose(y_pred))
    union = np.dot(y_true, np.transpose(y_true)) + np.dot(y_pred, np.transpose(y_pred))
    return np.mean((2. * intersection + smooth) / (union + smooth))

def dice(im1, im2, empty_score=1.0):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score
        
    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum

###########################################################################################################################

'''
Defining total number of classes and 2D image Dimensions
'''
nb_classes = 2
img_w = 256
img_h = 128

#########################################################################################################
###########################################################################################################
#Define the neural network
def getNet(patchHeight, patchWidth,patchDepth,  ipCh, outCh):
    # Input
    input1 = Input((patchHeight, patchWidth, patchDepth, 1))
    # Encoder
    conv0 = ZeroPadding3D(padding=(3, 3, 0))(input1)
    conv0 = Conv3D(16, (7, 7, patchDepth))(conv0)
    conv0 = BatchNormalization()(conv0)
    conv0 = Activation('relu')(conv0)
    conv0 = Dropout(0.2)(conv0)
    
    #
    ReShp = Reshape((patchHeight,patchWidth, 16))(conv0)
    conv1 = Conv2D(16, (3, 3), padding='same')(ReShp)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Dropout(0.2)(conv1)

    #
    conv1 = concatenate([ReShp, conv1], axis=-1)
    conv1 = Conv2D(16, (3, 3), padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    #
    input2 = MaxPooling2D(pool_size=(2, 2))(ReShp)
    conv21 = concatenate([input2, pool1], axis=-1)

    #
    conv2 = Conv2D(32, (3, 3), padding='same')(conv21)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Dropout(0.2)(conv2)

    #
    conv2 = concatenate([conv21, conv2], axis=-1)
    conv2 = Conv2D(32, (3, 3), padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    #
    input3 = MaxPooling2D(pool_size=(2, 2))(input2)
    conv31 = concatenate([input3, pool2], axis=-1)

    conv3 = Conv2D(64, (3, 3), padding='same')(conv31)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Dropout(0.2)(conv3)

    #
    conv3 = concatenate([conv31, conv3], axis=-1)
    conv3 = Conv2D(64, (3, 3), padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    input4 = MaxPooling2D(pool_size=(2, 2))(input3)
    conv41 = concatenate([input4, pool3], axis=-1)

    conv4 = Conv2D(128, (3, 3), padding='same')(conv41)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Dropout(0.2)(conv4)

    conv4 = concatenate([conv41, conv4], axis=-1)
    conv4 = Conv2D(128, (3, 3), padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Dropout(0.2)(conv4)

    conv4 = Conv2D(128, (3, 3), padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)

    # Decoder
    conv5 = UpSampling2D(size=(2, 2))(conv4)
    conv51 = concatenate([conv3, conv5], axis=-1)

    conv5 = Conv2D(64, (3, 3), padding='same')(conv51)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Dropout(0.2)(conv5)

    conv5 = concatenate([conv51, conv5], axis=-1)
    conv5 = Conv2D(64, (3, 3), padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)

    #
    conv6 = UpSampling2D(size=(2, 2))(conv5)
    conv61 = concatenate([conv2, conv6], axis=-1)

    conv6 = Conv2D(32, (3, 3), padding='same')(conv61)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Dropout(0.2)(conv6)

    conv6 = concatenate([conv61, conv6], axis=-1)
    conv6 = Conv2D(32, (3, 3), padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)

    #
    conv7 = UpSampling2D(size=(2, 2))(conv6)
    conv71 = concatenate([conv1, conv7], axis=-1)

    conv7 = Conv2D(16, (3, 3), padding='same')(conv71)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Dropout(0.2)(conv7)

    conv7 = concatenate([conv71, conv7], axis=-1)
    conv7 = Conv2D(16, (3, 3), padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)

    # Final
    conv81 = UpSampling2D(size=(8, 8))(conv4)
    conv82 = UpSampling2D(size=(4, 4))(conv5)
    conv83 = UpSampling2D(size=(2, 2))(conv6)
    conv8 = concatenate([conv81, conv82, conv83, conv7], axis=-1)
    conv8 = Conv2D(outCh, (1, 1), activation='softmax')(conv8)

    ############
    model = Model(inputs=input1, outputs=conv8)
    adm = Adam(lr=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-08)
    sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(optimizer=adm, loss=weighted_pixelwise_crossentropy(1))
    model.compile(optimizer=adm, loss=losses.categorical_crossentropy)

    return model


# ipDepth = 256
# ipHeight = 256
# ipWidth = 128
# ipCh = 1
# # ipHeight =2848
# # ipWidth = 4288


# outDepth = 256

# # Get OD segmentation Net
# odSegNet = getNet(ipHeight, ipWidth, ipDepth, ipCh, outDepth)
# print "---------------- Cerebellum segmentation network ----------------"
# odSegNet.summary()

# # Read train, val and test numpy
# load_dir = "./Data/"
# train_x = np.load(load_dir + "train_x.npy")
# train_y = np.load(load_dir + "train_y.npy")
# val_x = np.load(load_dir + "val_x.npy")
# val_y = np.load(load_dir + "val_y.npy")
# test_x = np.load(load_dir + "test_x.npy")
# test_y = np.load(load_dir + "test_y.npy")
# # sflag = 0
# # saveLossHist = "./output/"+ modelName + str(sEpoch) + "_" + str(lrate) + ".txt"
# # loadWeightsPath = "./weights/" + modelName + str(sEpoch-1) + ".hdf5"

# # if not os.path.exists("./output/"+ modelName):
# #     os.makedirs("./output/"+ modelName)
# # if not os.path.exists("./weights/"+ modelName):
# #     os.makedirs("./weights/"+ modelName)
# # if not os.path.exists("./model/"+ modelName):
# #     os.makedirs("./model/"+ modelName)

# # saveWeightsPath = "./weights/" + modelName + str(i) + ".hdf5"
# # saveWeights = ModelCheckpoint(filepath=saveWeightsPath)

# odSegNet.fit(train_x, train_y, epochs=1,batch_size=1,shuffle=True)

# # lossHistory = odSegNet.history.history['loss']
# # lossArray = np.array(lossHistory)
# # with open(saveLossHist, 'a') as f:
# # 	np.savetxt(f, lossArray, delimiter=",")

# # Save model
# # saveModelPath = "./model/" + modelName + str(i) + ".h5"
# # odSegNet.save(saveModelPath)


############################################################################################
#############################################################################################

ipDepth = 51
ipHeight = 256
ipWidth = 128
ipCh = 1
# ipHeight =2848
# ipWidth = 4288


outDepth = 2

# Get Cerebellum segmentation Net
CerSegNet = getNet(ipHeight, ipWidth, ipDepth, ipCh, outDepth)
print "---------------- Cerebellum segmentation network ----------------"
CerSegNet.summary()

'''
Some variable which will be used
'''
tr_images = 2     # NUmber of Training  Images
valid_images = 2  # Number of validation Images
test_images = 2   # Number of Testing Images
no_ep = 30        # Total Number of Epochs   
ep_lap = 1
weig_load = 0     # weight load epoch number
b_size = 8        # batch size
best_v_dice = 0   # best validation dice
best_epoch = 0    # epoch number at which best validation dice was achieved

main_path ='./' 
log_file = main_path + "Log/log.txt"

if weig_load != 0: 
    graph.load_weights(main_path+'CNN_output/weights/weights-%d.hdf5'%(weig_load))
    print('\n-----------------------Loading Weight %d---------------------------\n'%(weig_load))

############################################################################
'''
Load whole training data at one go
'''
training_data = np.load('./Data/train.npz')
print training_data.files
training_inp = training_data['x'].astype('float32')
training_gt = training_data['y'].astype('uint8')[:,:,:,0]
# training_inp = np.transpose(training_inp)
# training_inp = training_inp.reshape(training_inp.shape+(1,)).astype('float32')
# training_gt = np.transpose(training_gt)

print(training_inp.shape)
print(training_gt.shape)

training_gt_category = my_to_categorical(training_gt, nb_classes)
print('training GT after categorical')
print(training_gt_category.shape)

if(np.all(training_gt == np.argmax(training_gt_category,axis=-1))):
    print('Proper Conversion to categorical')

############################################################################### 
###################################################################################

for ep in range(weig_load+1,no_ep+1):
    print('\n\n###########################################')
    print('##########################################\n\n')

    print('Training Model:\n\n')
    print('Epoch: %d' %(ep))
    
    MCP=keras.callbacks.ModelCheckpoint(main_path+'CNN_output/weights/weights-%d.hdf5'%(ep), monitor='val_loss',save_best_only=False)
    
    CerSegNet.fit(training_inp , training_gt_category, batch_size=b_size,  verbose=1 ,epochs=1, callbacks = [MCP])
    
    
######################################################################################################
    
    if((ep % ep_lap) == 0):

        print('\n\n-----------------------------------------\n\n')
        print('Training Accuracy:')
        t_dice = 0.0
        t_acc = 0.0
        scr = 0.0    
        for trai_imgn in range(tr_images):
            print('\n\nloading image %d for Accuracy\n'% (trai_imgn))
            
            npz_contents = np.load(main_path+'Data/train/%d.npz'%(trai_imgn))
            print('data loaded')
            trai_inp = npz_contents['x'].astype('float32')
            
            trai_gt = npz_contents['y'].astype('uint8')
            # trai_inp = np.transpose(trai_inp)
            # trai_inp = trai_inp.reshape(trai_inp.shape+(1,)).astype('float32')
            # trai_gt = np.transpose(trai_gt)
            print (trai_inp.shape)
            
            prediction = CerSegNet.predict(trai_inp, batch_size=b_size)

            trai_pre = np.argmax(prediction ,axis=-1).astype('uint8')
            print trai_pre.shape
            print trai_gt.shape, trai_pre.shape
            trai_pre = np.reshape(trai_pre,trai_pre.shape[0]*trai_pre.shape[1]*trai_pre.shape[2]).astype('uint8')
            trai_gt = np.reshape(trai_gt,trai_gt.shape[0]*trai_gt.shape[1]*trai_gt.shape[2]).astype('uint8')

            [my_accu,zero_accu] = accuracy(trai_pre, trai_gt)
            # skl_dice = f1_score(trai_gt, trai_pre,average='macro')
            skl_dice = dice(trai_gt, trai_pre)
            skl_accu = accuracy_score(trai_gt, trai_pre)

            print ('skl accu = ',skl_accu,'skl dice coeff = ',skl_dice,'zero accu = ',zero_accu, 'my_accu = ', my_accu)

            print "LOLOLOL"
            trai_pre = np.reshape(trai_pre,[trai_inp.shape[0],trai_inp.shape[1],trai_inp.shape[2]]).astype('uint8')

            new_image = nib.Nifti1Image(np.swapaxes(np.swapaxes(trai_pre, 0 , 2),0,1), affine=np.eye(4))

            print new_image.shape
            new_image.to_filename(main_path+'test.nii.gz')

            score = 0
            t_dice+=skl_dice
            t_acc += skl_accu
            scr=scr+score
            fp1 = open(log_file,'a')
            fp1.write('epoch:%d image:%d Training accuracy:%f Dice Coeff:%f\n'%(ep,trai_imgn,skl_accu,skl_dice))
            fp1.close()
            

        scr = scr/tr_images
        t_dice = t_dice/tr_images
        t_acc = t_acc/tr_images
        print('\n\nTraining Overall Dice Coeffient: %f'%(t_dice))
        print('Training Overall Accuracy: %f'%(t_acc))
        print('Training Score: %f'%(scr))
            
    ######################################################################################################
        
        print('\n-----------------------------------------\n\n')
        print('validation Accuracy:')
        t_dice = 0.0
        t_acc = 0.0
        scr =0.0    
        
        for valid_imgn in range(valid_images):
            print('\nloading image %d for Accuracy\n'% (valid_imgn))

            npz_contents = np.load(main_path+'Data/val/%d.npz'%(valid_imgn))
            print('data loaded')
            valid_inp = npz_contents['x'].astype('float32')
            valid_gt = npz_contents['y'].astype('uint8')
            # valid_inp = np.transpose(valid_inp)
            # valid_inp = valid_inp.reshape(valid_inp.shape+(1,)).astype('float32')
            # valid_gt = np.transpose(valid_gt)
            
            print (valid_inp.shape)
            
            prediction = CerSegNet.predict(valid_inp , batch_size=b_size)    
            valid_pre = np.argmax(prediction ,axis=-1).astype('uint8')

            valid_pre = np.reshape(valid_pre,valid_pre.shape[0]*valid_pre.shape[1]*valid_pre.shape[2]).astype('uint8')
            valid_gt = np.reshape(valid_gt,valid_gt.shape[0]*valid_gt.shape[1]*valid_gt.shape[2]).astype('uint8')

            [my_accu,zero_accu] = accuracy(valid_pre, valid_gt)
            # skl_dice = f1_score(trai_gt, trai_pre,average='macro')
            skl_dice = dice(valid_gt, valid_pre)
            skl_accu = accuracy_score(valid_gt, valid_pre)

            print ('\nskl accu = ',skl_accu,'skl dice coeff = ',skl_dice,'zero accu = ',zero_accu,'my_accu = ',my_accu)
            valid_pre = np.reshape(valid_pre,[valid_inp.shape[0],valid_inp.shape[1],valid_inp.shape[2]]).astype('uint8')
            score = 0
            t_dice+=skl_dice
            t_acc += skl_accu
            scr=scr+score
            fp1 = open(log_file,'a')
            fp1.write('epoch:%d image:%d validation accuracy:%f Dice Coeff:%f\n'%(ep,valid_imgn,skl_accu,skl_dice))
            fp1.close()
            

        scr = scr/valid_images
        t_dice = t_dice/valid_images
        t_acc = t_acc/valid_images
        if(t_dice > best_v_dice):
            best_v_dice = t_dice
            best_epoch = ep
            print('\n\nvalidation Overall Dice Coeffient: %f'%(t_dice))
            print('validation Overall Accuracy: %f'%(t_acc))
            print('validation Score: %f'%(scr))
            
######################################################################################################
######################################################################################################


'''
Test architecture performance on model weight for which best validation dice was achieved 
'''

print('\n-----------------------------------------\n\n')
print('Testing Accuracy:')
t_dice = 0.0
t_acc = 0.0
scr =0.0    

CerSegNet.load_weights(main_path+'CNN_output/weights/weights-%d.hdf5'%(best_epoch))
print('\n-----------------------Loading Best Weight %d for testing---------------------------\n'%(best_epoch))

for test_imgn in range(test_images):
    print('\nloading image %d for Accuracy\n'% (test_imgn))
    npz_contents = np.load(main_path+'Data/test/%d.npz'%(test_imgn))
    print('data loaded')
    test_inp = npz_contents['x'].astype('float32')
    test_gt = npz_contents['y'].astype('uint8')
    # test_inp = np.transpose(test_inp)
    # test_inp = test_inp.reshape(test_inp.shape+(1,)).astype('float32')
    # test_gt = np.transpose(test_gt)
            
    print (test_inp.shape)
            
    prediction = CerSegNet.predict(test_inp, batch_size=b_size)

            
    test_pre = np.argmax(prediction,axis=-1).astype('uint8')

    test_pre = np.reshape(test_pre,test_pre.shape[0]*test_pre.shape[1]*test_pre.shape[2]).astype('uint8')
    test_gt = np.reshape(test_gt,test_gt.shape[0]*test_gt.shape[1]*test_gt.shape[2]).astype('uint8')

    [my_accu,zero_accu] = accuracy(test_pre, test_gt)
    # skl_dice = f1_score(trai_gt, trai_pre,average='macro')
    skl_dice = dice(test_gt, test_pre)
    skl_accu = accuracy_score(test_gt, test_pre)

    print ('\nskl accu = ',skl_accu,'skl dice coeff = ',skl_dice,'zero accu = ',zero_accu,'my_accu = ',my_accu)
    test_pre = np.reshape(test_pre,[test_inp.shape[0],test_inp.shape[1],test_inp.shape[2]]).astype('uint8')
    score = 0
    t_dice+=skl_dice
    t_acc += skl_accu
    scr=scr+score
    fp1 = open(log_file,'a')
    fp1.write('best epoch:%d image:%d test accuracy:%f Dice Coeff:%f\n'%(best_epoch,test_imgn,skl_accu,skl_dice))
    fp1.close()
    
scr = scr/test_images
t_dice = t_dice/test_images
t_acc = t_acc/test_images
print('\n\ntesting Overall Dice Coeffient: %f'%(t_dice))
print('testing Overall Accuracy: %f'%(t_acc))
print('testing Score: %f'%(scr))