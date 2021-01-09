import numpy as np
import keras
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score as score
import random


class MODEL:

    def __init__(self, xtrain, ytrain, xtest, ytest, load_model,
                 draw_model, show_summary, show_details, save_loss,
                 save_accuracy, learning_rate, batch_size, num_epochs):

        self.results_path = 'results/Adam_b128_e10_lr0.001/'

        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xtest = xtest
        self.ytest = ytest

        self.xtrain.shape = (-1,28,28,1)
        self.xtest.shape = (-1,28,28,1)

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.load_model = load_model
        self.save_loss = save_loss
        self.save_accuracy = save_accuracy
        self.draw_model = draw_model
        self.show_details = show_details
        self.show_summary = show_summary

        if self.load_model:
            self.load_my_model()
            if self.draw_model:
                keras.utils.plot_model(self.my_model,to_file='my_model.png',
                                       show_shapes=True,show_layer_names=True)
        else:
            self.build_model()
            if self.draw_model:
                keras.utils.plot_model(self.my_model,to_file='my_model.png',
                                       show_shapes=True,show_layer_names=True)
            self.fit()

    def load_my_model(self):
        self.my_model = keras.models.load_model(self.results_path+'my_model.hdf5')
        if self.show_summary:
            self.my_model.summary()

    def build_model(self):
        self.num_classes = int(np.unique(self.ytrain).shape[0])
        self.my_model = keras.models.Sequential()

        self.my_model.add(keras.layers.Conv2D(32, kernel_size=(
            3, 3), activation='relu', input_shape=self.xtrain.shape[1:],name='Conv_2D_1'))
        
        self.my_model.add(keras.layers.Conv2D(64, kernel_size=(
            3, 3), activation='relu',name='Conv_2D_2'))

        self.my_model.add(keras.layers.MaxPooling2D(pool_size=(2, 2),name='MaxPool'))
        
        self.my_model.add(keras.layers.Dropout(0.25,name='Dropout_1'))

        self.my_model.add(keras.layers.Flatten(name='Flatten'))
        
        self.my_model.add(keras.layers.Dense(128, activation='relu',name='Dense'))
        
        self.my_model.add(keras.layers.Dropout(0.5,name='Dropout_2'))

        self.my_model.add(keras.layers.Dense(
            self.num_classes, activation='softmax',name='Output_Dense_2'))

        self.my_model.compile(loss=keras.losses.categorical_crossentropy,
                              optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                              metrics=['accuracy'])

        if self.show_summary:
            self.my_model.summary()

    def fit(self):
        self.new_ytrain = keras.utils.to_categorical(
            self.ytrain, self.num_classes)
        self.new_ytest = keras.utils.to_categorical(
            self.ytest,self.num_classes)

        self.history = self.my_model.fit(self.xtrain,
                                         self.new_ytrain,
                                         batch_size=self.batch_size,
                                         epochs=self.num_epochs,
                                         verbose=self.show_details,
                                         validation_data=(self.xtest,self.new_ytest))

        self.my_model.save(self.results_path+'my_model.hdf5')

        self.my_training_loss = self.history.history['loss']
        self.my_training_accuracy = self.history.history['accuracy']
        self.my_validation_loss = self.history.history['val_loss']
        self.my_validation_accuracy = self.history.history['val_accuracy']

        plt.plot(self.my_training_loss, color='blue', label='train_loss')
        plt.plot(self.my_validation_loss,color='red',label='val_loss')
        plt.legend()
        plt.savefig(self.results_path+'my_loss.png')
        plt.clf()
        plt.plot(self.my_training_accuracy, color='blue', label='train_acc')
        plt.plot(self.my_validation_accuracy,color='red',label='val_acc')
        plt.legend()
        plt.savefig(self.results_path+'my_accuracy.png')

    def evaluate_score(self):
        self.ypred = self.my_model.predict(self.xtest)
        self.ypred = np.argmax(self.ypred, axis=1)
        assert(self.ypred.shape == self.ytest.shape)
        return score(self.ypred,self.ytest)
        
    
    def predict_test_image(self,img):
        img.shape = (1,28,28,1)
        prediction = np.argmin(self.my_model.predict(img),axis=1)
        return prediction