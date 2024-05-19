import numpy as np
import pandas as pd 


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df_train= pd.read_csv('/kaggle/input/regularization-techniques/train.csv')
df_test = pd.read_csv('/kaggle/input/regularization-techniques/test.csv')

train_values=df_train.values
test_values = df_test.values

X_train = np.array(train_values[:,1:])
Y_train = np.array(train_values[:,0])
X_test = np.array(test_values[:,1:])
Y_test = np.array(test_values[:,0])

X_train=X_train.reshape(-1,28,28,1)/255.0
X_train.shape

X_test = X_test.reshape(-1,28,28,1)/255.0
X_test.shape

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from keras.utils.np_utils import to_categorical 

number_of_labels = np.max(Y_train)
number_of_labels

def create_model(INPUT_SHAPE):
    model = Sequential()
    model.add(Conv2D(filters = 8, kernel_size=(3, 3),strides=(1,1),padding="valid", input_shape=INPUT_SHAPE))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding="same"))

    model.add(Conv2D(filters = 16, kernel_size=(3, 3),strides=(1,1),padding="valid"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding="same"))

    model.add(Conv2D(filters = 32, kernel_size=(3, 3),strides=(1,1),padding="valid"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(number_of_labels+1, activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.summary()
    return model

X_train[0].shape

model=create_model(X_train[0].shape)

to_categorical(Y_train,number_of_labels+1).shape

model.fit(X_train,to_categorical(Y_train,number_of_labels+1),batch_size=32,epochs=5,validation_split=0.2)

loss,accuracy=model.evaluate(X_test,to_categorical(Y_test,number_of_labels+1))
print("Accuracy of the model without regularization: "+ str(accuracy))

from keras import regularizers  

def create_model_with_L1(INPUT_SHAPE):
    model = Sequential()
    model.add(Conv2D(filters = 8, kernel_size=(3, 3),strides=(1,1),padding="valid", input_shape=INPUT_SHAPE))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding="same"))

    model.add(Conv2D(filters = 16, kernel_size=(3, 3),strides=(1,1),padding="valid"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding="same"))

    model.add(Conv2D(filters = 32, kernel_size=(3, 3),strides=(1,1),kernel_regularizer=regularizers.l2(0.01),padding="valid"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Flatten())
    model.add(Dense(128,kernel_regularizer=regularizers.l1(0.01), activation='relu'))
    model.add(Dense(128,kernel_regularizer=regularizers.l1(0.01), activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64,kernel_regularizer=regularizers.l1(0.01), activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32,kernel_regularizer=regularizers.l1(0.01), activation='relu'))
    model.add(Dense(number_of_labels+1, activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.summary()
    return model


model=create_model_with_L1(X_train[0].shape)

model.fit(X_train,to_categorical(Y_train,number_of_labels+1),batch_size=32,epochs=5,validation_split=0.2)             

loss,accuracy=model.evaluate(X_test,to_categorical(Y_test,number_of_labels+1))
print("Accuracy of the model with L1 Regularization: "+ str(accuracy))

def create_model_with_L2_Dropout(INPUT_SHAPE):
    model = Sequential()
    model.add(Conv2D(filters = 8, kernel_size=(3, 3),strides=(1,1),padding="valid", input_shape=INPUT_SHAPE))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding="same"))

    model.add(Conv2D(filters = 16, kernel_size=(3, 3),strides=(1,1),padding="valid"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding="same"))

    model.add(Conv2D(filters = 32, kernel_size=(3, 3),strides=(1,1),kernel_regularizer=regularizers.l2(0.01),padding="valid"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    
    model.add(Flatten())
    model.add(Dense(128,kernel_regularizer=regularizers.l2(0.01), activation='relu'))
    model.add(Dense(128,kernel_regularizer=regularizers.l2(0.01), activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64,kernel_regularizer=regularizers.l2(0.01), activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32,kernel_regularizer=regularizers.l2(0.01), activation='relu'))
    model.add(Dense(number_of_labels+1, activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.summary()
    return model

model=create_model_with_L2_Dropout(X_train[0].shape)

model.fit(X_train,to_categorical(Y_train,number_of_labels+1),batch_size=32,epochs=5,validation_split=0.2)

loss,accuracy=model.evaluate(X_test,to_categorical(Y_test,number_of_labels+1))
print("Accuracy of the model with L2 regularization and Dropout: "+ str(accuracy))

def create_model_with_L1_L2_Dropout_EarlyStopping(INPUT_SHAPE):
    model = Sequential()
    model.add(Conv2D(filters = 8, kernel_size=(3, 3),strides=(1,1),padding="valid", input_shape=INPUT_SHAPE))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding="same"))

    model.add(Conv2D(filters = 16, kernel_size=(3, 3),strides=(1,1),padding="valid"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding="same"))

    model.add(Conv2D(filters = 32, kernel_size=(3, 3),strides=(1,1),kernel_regularizer=regularizers.l1(0.01),padding="valid"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    
    model.add(Flatten())
    model.add(Dense(128,kernel_regularizer=regularizers.l1(0.01), activation='relu'))
    model.add(Dense(128,kernel_regularizer=regularizers.l2(0.01), activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64,kernel_regularizer=regularizers.l2(0.01), activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32,kernel_regularizer=regularizers.l2(0.01), activation='relu'))
    model.add(Dense(number_of_labels+1, activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.summary()
    return model


model=create_model_with_L1_L2_Dropout_EarlyStopping(X_train[0].shape)

from keras.optimizers import RMSprop,SGD,Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

checkpoint = ModelCheckpoint(r'C:\Label.h5',
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             verbose=1)

earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=3,
                          verbose=1,
                          restore_best_weights=True
                          )

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=3,
                              verbose=1,
                              min_delta=0.0001)

callbacks = [earlystop,checkpoint,reduce_lr]


model.fit(X_train,to_categorical(Y_train,number_of_labels+1),
          batch_size=32,epochs=10,
          validation_split=0.2,
          callbacks=callbacks)


loss,accuracy=model.evaluate(X_test,to_categorical(Y_test,number_of_labels+1))
print("Accuracy of the model with L2 , L1 and Dropout with earlyStopping and ReduceLrOnPlateau: "+ str(accuracy))

X_tra=X_train.reshape(-1,784)
X_tes=X_test.reshape(-1,784)

from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier(random_state=1)
model.fit(X_tra,Y_train)

model.score(X_tes,Y_test)

from xgboost import XGBClassifier
model_xgb = XGBClassifier()
model_xgb.fit(X_tra, Y_train)
model_xgb.score(X_tes,Y_test)                 
