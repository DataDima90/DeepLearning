# Import internal packages
from base.base_data_loader import BaseDataLoader

# Import external packages
from keras.datasets import mnist
from keras.utils import to_categorical


class DataLoader(BaseDataLoader):
    def __init__(self, config):
        super(DataLoader, self).__init__(config)
        (self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()
        self.X_train = self.X_train.reshape((self.X_train.shape[0], 28, 28, 1))
        self.X_test = self.X_test.reshape((self.X_test.shape[0], 28, 28, 1))
        self.y_train = to_categorical(self.y_train)
        self.y_test = to_categorical(self.y_test)
        self.X_train_norm, self.X_test_norm = self.prep_pixels()

    def prep_pixels(self):
        """Scale pixels between 0 and 1"""

        # convert from integers to floats
        train_norm = self.X_train.astype('float32')
        test_norm = self.X_test.astype('float32')

        # normalize to range between 0 and 1
        X_train_norm = train_norm / 255.0
        X_test_norm = test_norm / 255.0

        return X_train_norm, X_test_norm

    def get_train_data(self):
        return self.X_train_norm, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test
