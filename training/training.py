# Import internal packages
from base.base_training import BaseTraining

# Import external packages
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
import os


def create_model():
    # Import external packages
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.optimizers import Adam

    model = Sequential()
    model.add(Dense(10, input_shape=(4,), activation='tanh'))
    model.add(Dense(8, activation='tanh'))
    model.add(Dense(6, activation='tanh'))
    model.add(Dense(3, activation='softmax'))

    opt = Adam(lr=0.01)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy'])

    return model


class ModelTraining(BaseTraining):
    def __init__(self, model, data, config):
        super(ModelTraining, self).__init__(model, data, config)
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

    def cross_validation(self):
        """"""
        estimator = KerasClassifier(build_fn=self.model,
                                    epochs=self.config.training.num_epochs,
                                    verbose=self.config.training.verbose_training,
                                    batch_size=self.config.training.batch_size)

        cv_scores = cross_val_score(estimator=estimator,
                                    X=self.data[0],
                                    y=self.data[1],
                                    cv=2)

        print("Accuracy : {:0.2f} (+/- {:0.2f}}".format(cv_scores.mean(), cv_scores.std()))

    def hyp_opt(self):
        estimator = KerasClassifier(build_fn=create_model,
                                    epochs=self.config.training.num_epochs,
                                    verbose=self.config.training.verbose_training,
                                    batch_size=self.config.training.batch_size)

        batch_size = [10, 20, 40, 60, 80, 100]
        epochs = [10, 50, 100]
        param_grid = dict(batch_size=batch_size, epochs=epochs)

        grid = GridSearchCV(estimator=estimator,
                            param_grid=param_grid,
                            n_jobs=-1,
                            cv=3)

        grid_result = grid.fit(X=self.data[0], y=self.data[1])

        # summarize results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']

        for mean, stddev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stddev, param))
