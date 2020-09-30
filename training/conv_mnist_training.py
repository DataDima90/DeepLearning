# Import internal packages
from base.base_training import BaseTraining

# Import external packages
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger
import os


class ConvMnistModelTraining(BaseTraining):
    def __init__(self, model, data, config):
        super(ConvMnistModelTraining, self).__init__(model, data, config)
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.init_callbacks()

    def init_callbacks(self):
        """
        Initialize callbacks for training the model
        :return:
        """
        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(self.config.callbacks.checkpoint_dir,
                                      "%s-{}".format(
                                          self.config.callbacks.checkpoint_filepath[1]) % self.config.exp.name),
                monitor=self.config.callbacks.checkpoint_monitor,
                mode=self.config.callbacks.checkpoint_mode,
                save_best_only=self.config.callbacks.checkpoint_save_best_only,
                save_weights_only=self.config.callbacks.checkpoint_save_weights_only,
                verbose=self.config.callbacks.checkpoint_verbose
            )
        )

        self.callbacks.append(
            CSVLogger(self.config.callbacks.checkpoint_dir + "/log.csv", separator=",", append=False)
        )

        self.callbacks.append(
            TensorBoard(
                log_dir=self.config.callbacks.tensorboard_log_dir,
                write_graph=self.config.callbacks.tensorboard_write_graph
            )
        )

    def train(self):
        """
        Train the model
        :return:
        """

        history = self.model.fit(
            self.data[0], self.data[1],
            epochs=self.config.training.num_epochs,
            verbose=self.config.training.verbose_training,
            batch_size=self.config.training.batch_size,
            validation_split=self.config.training.validation_split,
            callbacks=self.callbacks
        )

        self.loss.extend(history.history['loss'])
        self.acc.extend(history.history['accuracy'])
        self.val_loss.extend(history.history['val_loss'])
        self.val_acc.extend(history.history['val_accuracy'])
