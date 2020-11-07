# Import external packages
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_history(config):
    """Plotting Accuracy and Loss Learning Curves for the Baseline Model"""
    print('experiments/' + config.evaluation.date + '/' + config.exp.name + '/checkpoints/log.csv')
    history = pd.read_csv('experiments/' + config.evaluation.date + '/' + config.exp.name + '/checkpoints/log.csv')
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


def plot_roc(model, data):
    """Plotting ROC Curve"""
    y_prediction = model.predict(data[0])
    fpr, tpr, threshold = roc_curve(data[1].ravel(), y_prediction.ravel())

    plt.figure(figsize=(10, 10))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label="AUC = {:.3f}".format(auc(fpr, tpr)))

    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC Curve")
    plt.legend()

    plt.show()


def evaluate(model, data):

    loss, accuracy = model.evaluate(data[0], data[1], verbose=False)

    print("Accuracy: {:.4f}".format(accuracy))
    print("Loss: {:.4f}".format(loss))


def classification_accuracy_report(model, data):
    """Accuracy of the predicted values"""
    y_prediction = model.predict(data[0])

    print("Classification Report")
    print(classification_report(y_true=np.argmax(data[1], axis=1),
                                y_pred=np.argmax(y_prediction, axis=1)))

    print("Confusion Matrix")
    print(confusion_matrix(y_true=np.argmax(data[1], axis=1),
                           y_pred=np.argmax(y_prediction, axis=1)))

