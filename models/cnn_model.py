# Import internal packages
from base.base_model import BaseModel

# Import external packages
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten
from keras.optimizers import SGD


class ConvModel(BaseModel):
    """Convolutional Neural Network Architecture"""

    def __init__(self, config):
        super(ConvModel, self).__init__(config)
        self.build_model()

    def build_model(self):
        """
        CNN model with MaxPool2D and Relu as a filter, Dropout for regularization and SGD as an optimizer
        """
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3),
                              activation='relu', input_shape=(28, 28, 1)))
        self.model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(10, activation='softmax'))

        opt = SGD(lr=0.01, momentum=0.9)

        self.model.compile(
            loss='binary_crossentropy',
            optimizer=opt,
            metrics=['accuracy'])

        return self.model
