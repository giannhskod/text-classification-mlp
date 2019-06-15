from keras.callbacks import ModelCheckpoint
from keras_tqdm import TQDMNotebookCallback

from app.metrics import f1, accuracy

def load_model(x_train, y_train, x_train_dev, y_train_dev, kwargs):
    """

    :param x_train:
    :param y_train:
    :param x_train_dev:
    :param y_train_dev:
    :param kwargs:
    :return:
    """
    from keras.models import Sequential
    from keras.layers import Dropout, Dense
    from talos.model import early_stopper

    n_labels = y_train.shape[1]
    visualize_process = kwargs.get('visualize_process')
    model = Sequential()
    model.add(Dense(units=kwargs.get('first_neuron', 2),
                    input_dim=x_train.shape[1],
                    activation=kwargs.get('activation', 'relu')))

    # Dropout probability in order to avoid overfitting.
    model.add(Dropout(kwargs.get('dropout', 0.5)))

    #1st Hidden Layer
    default_hidden_layer_units = kwargs.get('first_neuron', 2) / 2
    model.add(Dense(units=kwargs.get('first_hidden_layer', default_hidden_layer_units),
                    activation=kwargs.get('activation', 'relu')))
    model.add(Dropout(kwargs.get('dropout', 0.5)))

    # 2nd & last Hidden layer
    # Mutual exclusive Classes
    model.add(Dense(n_labels, activation='softmax'))

    default_callbacks = [early_stopper(kwargs['epochs'], mode='moderate', monitor='val_fmeasure')]
    if visualize_process:
        print(model.summary())
        checkpoint = ModelCheckpoint(kwargs.get('model_type', 'keras_tf_idf_model'),
                                     monitor='val_accuracy',
                                     verbose=1,
                                     save_best_only=True,
                                     mode='max')

        checkpoint2 = ModelCheckpoint(kwargs.get('model_type', 'keras_tf_idf_model'),
                                     monitor='val_f1',
                                     verbose=1,
                                     save_best_only=True,
                                     mode='max')

        extra_callbacks = [checkpoint, TQDMNotebookCallback(), checkpoint2]
    else:
        extra_callbacks = []

    # Model compilation parameterized with
    model.compile(loss='categorical_crossentropy',
                  optimizer=kwargs.get('optimizer', 'Adam'),
                  metrics=[f1, accuracy])

    history = model.fit(x_train,y_train,
                        batch_size=kwargs.get('batch_size', 32),
                        epochs=kwargs.get('epochs', 10),
                        verbose=0,
                        callbacks=default_callbacks + extra_callbacks,
                        validation_data=(x_train_dev, y_train_dev),
                        shuffle=True)

    return history, model