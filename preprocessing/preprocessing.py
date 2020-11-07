# Import external packages
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from dataloader.iris_data_loader import IrisDataLoader


#class Preprocessing(IrisDataLoader):
#    def __init__(self, config):
#        super(IrisDataLoader, self).__init__(config)
#        self.dl = IrisDataLoader(config)
#        self.X_train, self.y_train = self.dl.get_train_data()


if __name__ == '__main__':
    from sklearn import datasets

    iris = datasets.load_iris()
    X = iris['data']
    features_names = iris['feature_names']

    print(features_names)
    print(iris)