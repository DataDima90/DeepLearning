# Import internal packages
from base.base_data_loader import BaseDataLoader

# Import external packages
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


class IrisDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(IrisDataLoader, self).__init__(config)
        self.iris = datasets.load_iris()
        self.X = self.iris['data']
        self.y = self.iris['target']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.min_max_scaler(), self.encoder(), test_size=0.8, random_state=42)

    def encoder(self):
        encoder = OneHotEncoder()
        return encoder.fit_transform(self.y[:, np.newaxis]).toarray()

    def min_max_scaler(self):
        min_max_scaler = MinMaxScaler()
        return min_max_scaler.fit_transform(self.X)

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test
