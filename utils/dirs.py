import os


def create_dirs(dirs: list):
    """
    Create a list of directories if these directories can not be found
    :param dirs:
    :return: exit_code with 0:success or -1:failed
    """

    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)

        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)
