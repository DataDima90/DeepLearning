import matplotlib.pyplot as plt
import pandas as pd


def visualize(config):
    """Accuracy and Loss Learning Curves for the Baseline Model"""
    history = pd.read_csv(config.callbacks.checkpoint_dir + 'log.csv')
    epochs = range(len(history))

    plt.plot(epochs, history['accuracy'], 'bo', label='Training accuracy')
    plt.plot(epochs, history['val_accuracy'], 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, history['loss'], 'bo', label='Training loss')
    plt.plot(epochs, history['val_loss'], 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()
