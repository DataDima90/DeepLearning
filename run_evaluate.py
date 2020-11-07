from evaluation.evaluate import EvaluateConvMnistModel
from dataloader.mnist_data_loader import MnistDataLoader
from dataloader.iris_data_loader import IrisDataLoader
from models.cnn_model import ConvModel
from models.ann_model import ANNModel
from utils.config import process_config
from utils.utils import get_args
from utils.visualize import plot_history, plot_roc, classification_accuracy_report, evaluate

import logging


def main():
    """
    Capture the config path from the run arguments and then process the json configuration file
    :return:
    """

    config = None

    try:
        args = get_args()
        config = process_config(args.config)
        raise RuntimeError("Missing or invalid arguments")
    except Exception as e:
        logging.error("Failed", exc_info=e)

    print("Create the data generator.")
    # data_loader = MnistDataLoader(config=config)
    data_loader = IrisDataLoader(config=config)
    train_data = data_loader.get_train_data()
    test_data = data_loader.get_test_data()

    print("Build the model")
    # cnn_model = ConvModel(config=config).build_model()
    cnn_model = ANNModel(config=config).build_model()

    print("Load the best weights")
    cnn_model.load_weights("experiments/{}/{}/checkpoints/{}-weights.best.hdf5".format(
        config.evaluation.date, config.exp.name, config.exp.name))

    print("Evaluate the model")
    print("Training Metrics")
    evaluate(model=cnn_model, data=train_data)
    print("Testing Metrics")
    evaluate(model=cnn_model, data=test_data)

    # print("Visualize loss and accuracy for Training and Validation data")
    # plot_history(config=config)

    # print("Plotting ROC Curve")
    # plot_roc(model=cnn_model, data=test_data)

    print("Classifcation Accuracy Report")
    classification_accuracy_report(model=cnn_model, data=test_data)


if __name__ == '__main__':
    main()
