import tensorflow as tf
import numpy as np

class tensorclass():
    def __init__():
        mnist = tf.keras.datasets.mnist
        (x_train, y_train),(x_test , y_test) = mnist.load_data()
        x_train.shape

        x_train = tf.keras.utils.normalize(x_train , axis = 1)
        x_test = tf.keras.utils.normalize(x_test , axis = 1)


        img_size = 28
        x_trainer = np.array(x_train).reshape(-1,img_size,img_size,1)
        x_tester = np.array(x_test).reshape(-1,img_size,img_size,1)

        from tensorflow import keras
        model = keras.models.load_model('digit_recogniser_model.h5')