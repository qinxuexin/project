import keras
from keras.optimizers import adam
from CNN_model import bcnn
from data_processor import generator
from data_loader import x_train, y_train, x_vali, y_vali, num_class


def train_model(
        learning_rate=0.01,
        decay_learning_rate=1e-8,
        all_trainable=False,
        model_weights_path=None,
        no_class=197,
        batch_size=32,
        epoch=200):

    train_dict = {}
    for i, imgname in enumerate(x_train):
        train_dict[imgname] = y_train[i]

    vali_dict = {}
    for j, labname in enumerate(x_vali):
        vali_dict[labname] = y_vali[j]

    dict_data = {'train': train_dict, 'valid': vali_dict}

    train_ge = generator(dict_data, num_class, is_train=True)
    valid_ge = generator(dict_data, num_class, is_train=False)

    model = bcnn(
        all_trainable=all_trainable,
        no_class=no_class)

    model.summary()

    if model_weights_path:
        model.load_weights(model_weights_path)

    # Callbacks
    name_loss = 'categorical_crossentropy'
    optimizer = adam(lr=learning_rate, decay=decay_learning_rate)
    model.compile(loss=name_loss, optimizer=optimizer, metrics=['accuracy'])

    checkpoint = keras.callbacks.ModelCheckpoint('./weights.hdf5',
                                                 monitor='acc',
                                                 verbose=1,
                                                 save_best_only=True,
                                                 save_weights_only=True,
                                                 mode='auto')

    # Train
    history = model.fit_generator(
        generator=train_ge,
        validation_data=valid_ge,
        epochs=epoch,
        steps_per_epoch=len(dict_data['train']) // batch_size,
        validation_steps=len(dict_data['valid']) // batch_size,
        callbacks=[checkpoint],
        verbose=1)

    model.save_weights('./new_model_weights.h5')

    return history


if __name__ == "__main__":
    pass
