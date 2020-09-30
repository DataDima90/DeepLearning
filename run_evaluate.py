from evaluation.evaluate import EvaluateConvMnistModel
from dataloader.data_loader import DataLoader
from utils.config import process_config
from utils.utils import get_args
from utils.visualize import visualize

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
    data_loader = DataLoader(config=config)
    test_data = data_loader.get_test_data()

    print("Build the model")
    cnn_model = EvaluateConvMnistModel(config).build_model()

    print("Load the best weights")
    cnn_model.load_weights("experiments/{}/{}/checkpoints/{}-weights.best.hdf5".format(
        config.evaluation.date, config.exp.name, config.exp.name))

    print("Evaluate the model")
    scores = cnn_model.evaluate(test_data[0], test_data[1])
    print(scores)

    print("Visualize loss and accuracy for Training and Validation data")
    visualize(config=config)


if __name__ == '__main__':
    main()
