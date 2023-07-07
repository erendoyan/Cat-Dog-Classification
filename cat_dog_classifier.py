#Importing the neccessary libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D,Flatten,MaxPool2D,Dense
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.layers import BatchNormalization,Activation,Dropout
import matplotlib.pyplot as plt
from keras.optimizers import RMSprop

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)
test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)

train_generator = train_datagen.flow_from_directory(
    "cat_dog/training_set/training_set",
    target_size=(64, 64),
    class_mode="binary",
    batch_size=64,
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    "cat_dog/test_set/test_set",
    target_size=(64, 64),
    class_mode="binary",
    batch_size=64,
    shuffle=True
)

print(train_generator.class_indices)

for _ in range(5):
    img, label = train_generator.next()
    print(img.shape)
    plt.imshow(img[0])
    print(label[0])
    plt.show()

#Building the neural network
model = Sequential()

#First Layer
model.add(Conv2D(filters = 64,kernel_size=(3,3),data_format="channels_last",kernel_initializer="he_normal",input_shape=(64,64,3),strides=1))
model.add(BatchNormalization())
model.add(Activation("relu")) #Relu is commanly used in CNN algorithms
#62x62x64

#Second Layer
model.add(Conv2D(filters = 64,kernel_size=(3,3),strides=1))
model.add(BatchNormalization())
model.add(Activation("relu"))
#60x60x32
model.add(MaxPool2D(pool_size=(2,2),strides = 2))
model.add(Dropout(0.4)) #Dropout rate is 0.4
#30x30x32

#Third Layer
model.add(Conv2D(filters = 32,kernel_size = (3,3),strides = 1))
model.add(BatchNormalization())
model.add(Activation("relu"))
#28x28x32

#Fourth Layer
model.add(Conv2D(filters=32,kernel_size = (3,3),strides = 1))
model.add(BatchNormalization())
model.add(Activation("relu"))
#26x26x32

#Fifth Layer
model.add(Conv2D(filters=32,kernel_size = (3,3),strides = 1))
model.add(BatchNormalization())
model.add(Activation("relu"))
#24x24x32
model.add(MaxPool2D(pool_size=(2,2),strides = 2))
model.add(Dropout(0.4))
#12x12x32

#Sixth Layer
model.add(Flatten())
model.add(Dense(1024))
model.add(BatchNormalization())
model.add(Dense(128))
model.add(BatchNormalization())

#Last Layer
model.add(Dense(1))
model.add(Activation("sigmoid")) #Sigmoid func. used since it is binary classification

optimizer = RMSprop(learning_rate = 0.001)
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
model.summary() #Printing model summary

#Defining Callbacks
early_stop = EarlyStopping(monitor="val_accuracy",mode="max",min_delta=0.005,verbose=7,patience=5)
checkpoint = ModelCheckpoint(filepath="C:/Users/Talip Eren Doyan/Desktop/Deep Learning/CNN/cat_dog_classifier.h5",verbose=1,save_best_only=True)

#Training model with the train dataset
result = model.fit(train_generator,steps_per_epoch=62, epochs=50, verbose=2, validation_data=test_generator,callbacks=[checkpoint,early_stop],batch_size=128)

score = model.evaluate(test_generator)
test_generator.reset()

#Saving model
model.save("C:/Users/Talip Eren Doyan/Desktop/Deep Learning/CNN/cat_dog.h5")

#Defining accuracy and loss values and printing
acc = result.history["accuracy"]
print("Accuracy is :",acc[-1])
loss = result.history["loss"]
print("Loss is :",loss[-1])
val_acc = result.history["val_accuracy"]
print("Validation accuracy on test set is:",val_acc[-1])
val_loss = result.history["val_loss"]
print("Valdiation loss on the test set is:",val_loss[-1])

epoch = range(1,len(acc)+1)
plt.plot(epoch,acc,label="Eğitim Başarımı")
plt.plot(epoch,val_acc,label="Doğrulama Başarımı")
plt.title("Eğitim ve Doğrulama için Başarım")
plt.legend()

plt.figure()

plt.plot(epoch,loss,label="Eğitim Kaybı")
plt.plot(epoch,val_loss,label="Doğrulama Kaybı")
plt.title("Eğitim ve Doğrulama için Kayıp")
plt.legend()

plt.show()

