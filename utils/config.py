from dotmap import DotMap
import json
import time
import os


def get_config_from_json(json_file: str):
    """
    Get the config from a json file

    :param json_file:
    :return: config(namespace) or config(dictionary)
    """

    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    config = DotMap(config_dict)

    return config, config_dict


def process_config(json_file: str):
    """

    :param json_file: path to the json_file, e.g. config/conv_from_config.json
    :return: config(namespace)
    """
    config, _ = get_config_from_json(json_file)
    config.callbacks.tensorboard_log_dir = os.path.join("experiments",
                                                        time.strftime("%Y-%m-%d/", time.localtime()),
                                                        config.exp.name,
                                                        "logs/")
    config.callbacks.checkpoint_dir = os.path.join("experiments",
                                                   time.strftime("%Y-%m-%d/", time.localtime()),
                                                   config.exp.name,
                                                   "checkpoints/")

    return config
