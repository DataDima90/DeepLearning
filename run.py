# Import internal packages
from dataloader.data_loader import DataLoader
from models.cnn_model import ConvModel
from training.conv_mnist_training import ConvMnistModelTraining
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args


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
    data_loader = DataLoader(config=config)

    print("Create the model")
    model = ConvModel(config=config)

    print("Create Training environment")
    training = ConvMnistModelTraining(model=model.model,
                                      data=data_loader.get_train_data(),
                                      config=config)

    print("Start training the model")
    training.train()


if __name__ == '__main__':
    main()
