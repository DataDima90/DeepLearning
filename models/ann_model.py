# Import internal packages
from base.base_model import BaseModel

# Import external packages
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class ANNModel(BaseModel):
    """
    Fully connected Neural Network Architecture
    """

    def __init__(self, config):
        super(ANNModel, self).__init__(config)
        self.build_model()

    def build_model(self):
        """
        CNN model with Tanh as a filter and Adam as an optimizer
        """
        self.model = Sequential()
        self.model.add(Dense(10, input_shape=(4, ), activation='tanh'))
        self.model.add(Dense(8, activation='tanh'))
        self.model.add(Dense(6, activation='tanh'))
        self.model.add(Dense(3, activation='softmax'))

        opt = Adam(lr=0.01)

        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy'])

        # self.model.summary()

        return self.model
