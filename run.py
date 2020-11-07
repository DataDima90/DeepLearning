# Import internal packages
from dataloader.mnist_data_loader import MnistDataLoader
from dataloader.iris_data_loader import IrisDataLoader
from models.cnn_model import ConvModel
from models.ann_model import ANNModel
from training.training import ModelTraining
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV


def main():
    """
    Capture the config path from the run arguments and then process the json configuration file
    :return:
    """

    global config
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("Missing or invalid arguments")
        exit(0)

    # create the experiment dirs
    create_dirs(dirs=[config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir])

    print("Create the data generator.")
    # data_loader = MnistDataLoader(config=config)
    data_loader = IrisDataLoader(config=config)
    train_data = data_loader.get_train_data()

    print("Create the model")
    # model = ConvModel(config=config)
    model = ANNModel(config=config)

    print("Create Training environment")
    training = ModelTraining(model=model.model,
                             data=train_data,
                             config=config)

    print("Start training the model")
    # training.train()

    print("Start measuring performance with cross validation")
    training.hyp_opt()


if __name__ == '__main__':
    main()
