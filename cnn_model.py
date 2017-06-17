import keras
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.datasets import mnist
from keras.models import Sequential
from keras import backend as K


class ConvNet(object):
    """docstring for ConvNet."""
    def __init__(self, model):
        self.img_rows = 28
        self.img_cols = 28
        self.num_classes = 10
        self.model = model
        self.batch_size = 64
        self.epochs = 20

    def BuildCNN(self):
        self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=self.input_shape))
        self.model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.num_classes, activation='softmax'))

    def LoadData(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()

    def ReshapeData(self):
        if K.image_data_format() == 'channels_first':
            self.x_train = self.x_train.reshape(self.x_train.shape[0], 1, self.img_rows, self.img_cols)
            self.x_test = self.x_test.reshape(self.x_test.shape[0], 1, self.img_rows, self.img_cols)
            self.input_shape = (1, self.img_rows, self.img_cols)
        else:
            self.x_train = self.x_train.reshape(self.x_train.shape[0], self.img_rows, self.img_cols, 1)
            self.x_test = self.x_test.reshape(self.x_test.shape[0], self.img_rows, self.img_cols, 1)
            self.input_shape = (self.img_rows, self.img_cols, 1)

        self.x_train = self.x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')
        self.x_train /= 255
        self.x_test /= 255
        print('x_train shape:', self.x_train.shape)
        print(self.x_train.shape[0], 'train samples')
        print(self.x_test.shape[0], 'test samples')

        self.y_train = keras.utils.to_categorical(self.y_train, self.num_classes)
        self.y_test = keras.utils.to_categorical(self.y_test, self.num_classes)

    def print_shape(self):
        print('x_train shape: ', self.x_train.shape)
        print('y_train shape: ', self.y_train.shape)

    def trainCNN(self):
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                            optimizer=keras.optimizers.Adadelta(),
                            metrics=['accuracy'])
        self.model.fit(self.x_train, self.y_train,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       verbose=1,
                       validation_data=(self.x_test, self.y_test))

    def SaveModel(self):
        model_json = self.model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights("model.h5")


def main():
    model = Sequential()
    cnn = ConvNet(model)
    cnn.LoadData()
    print('Before reshaping training data dimensions:')
    cnn.print_shape()
    cnn.ReshapeData()
    print('After reshaping training data dimensions:')
    cnn.print_shape()
    cnn.BuildCNN()
    cnn.trainCNN()
    cnn.SaveModel()

if __name__ == '__main__':
    main()
