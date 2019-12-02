import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import math
import json

from src.data_generator import get_data_for_master_class
from src.siamese import get_siamese_layers, TripleGenerator

# solve plotting issues with matplotlib when no X connection is available
matplotlib.use('Agg')

classes = {
    # TODO: Add signs from belgium? change the class used?
    "RedRoundSign": {
        "signs_classes": ["p24", "w39", "pw3", "i9", "w51", "w59", "pl80", "i5", "pw4.2", "i7", "ph3", "ph4.2", "pe",
                          "i6", "il60", "w33", "p20", "w67", "ph2.6", "w62", "w60", "il100", "w23", "pl70", "ph4.8",
                          "w28", "ph4", "w45", "pr10", "w26", "w17", "pl40", "ph4.5", "w31", "pl0", "w5", "pw3.5",
                          "pa14", "p10", "pc", "w66", "pl90", "p28", "pa8", "p3", "p9", "pm8", "ph2.9", "p14", "ph4.3",
                          "p16", "p5", "ph5.5", "ph4.4", "pd", "w53", "pr80", "w54", "pa13", "p27", "w65", "pl10", "pg",
                          "ip", "p25", "pr100", "pl35", "il90", "ph3.3", "pw2", "il80", "pr50", "p29", "p1", "p12",
                          "w13", "p4", "w25", "w42", "w38", "p18", "ph2.4", "ph2.1", "pl60", "w10", "pl5", "w47", "p8",
                          "pm40", "w36", "w1", "pm46", "i15", "w29", "pm50", "pm5", "w48", "w19", "i1", "p22", "w57",
                          "pw3.2", "w56", "w3", "pn", "pm10", "w58", "w20", "ph1.5", "w2", "pl120", "p2", "p21", "p15",
                          "pr60", "pm15", "pb", "ph2", "pr30", "pm2", "pnl", "pw4.5", "w7", "w30", "ph3.8", "p6", "p11",
                          "p19", "i12", "pm35", "w55", "w52", "w37", "pl20", "pw2.5", "w41", "ph2.2", "w32", "pl15",
                          "pa12", "pm13", "pr20", "pa10", "p17", "w4", "w11", "pl65", "w64", "ps", "pw4", "pr70", "p13",
                          "w22", "ph2.5", "i10", "w9", "pn40", "pm30", "pl30", "w44", "pm55", "ph5.3", "w34", "i4",
                          "pr45", "i14", "pm20", "w21", "ph3.5", "w6", "il70", "w18", "ph2.8", "i3", "il50", "w8",
                          "p23", "w16", "p26", "w27", "pl110", "w12", "w24", "w63", "w15", "pm1.5", "w61", "i2", "pl50",
                          "w14", "il110", "pr40", "ph5", "pl25", "i8", "w46", "pl4", "i13", "w40", "pm2.5", "i11",
                          "w35", "w49", "pne", "w50", "p7", "pl100", "w43", "pl3", "ph3.2"],
        "merge_sign_classes": {
            # "prx": ['pr10', 'pr100', 'pr20', 'pr30', 'pr40', 'pr45', 'pr50', 'pr60', 'pr70', 'pr80', 'prx'],
            # "pmx": ['pm1.5', 'pm10', 'pm13', 'pm15', 'pm2', 'pm2.5', 'pm20', 'pm25', 'pm30', 'pm35', 'pm40', 'pm46',
            #         'pm5', 'pm49', 'pm50', 'pm55', 'pm8'],
            # "phx": ['ph', 'ph1.5', 'ph2', 'ph2.1', 'ph2.2', 'ph2.4', 'ph2.5', 'ph2.6', 'ph2.8', 'ph2.9', 'ph3.x', 'ph3',
            #         'ph3.2', 'ph3.3', 'ph3.5', 'ph3.7', 'ph3.8', 'ph38', 'ph39', 'ph45', 'ph4', 'ph4.2', 'ph4.3',
            #         'ph4.4', 'ph4.5', 'ph4.6', 'ph4.8', 'ph5', 'ph5.3', 'ph5.5', 'ph6'],
            # "pax": ['pa10', 'pa12', 'pa13', 'pa14', 'pa8', 'pax'],
            # "plo": ['pl35', 'pl25', 'pl15', 'pl10', 'pl110', 'pl65', 'pl90'],
            # "p_other": ['pw2', 'pw2.5', 'pw3', 'pw3.2', 'pw3.5', 'pw4', 'pw4.2', 'pw4.5',
            #             'p_prohibited_two_wheels_vehicules', 'p_prohibited_bicycle_and_pedestria',
            #             'p_prohibited_bicycle_and_pedestrian_issues', 'p13', 'p15', 'p16', 'p17', 'p18', 'p2', 'p21',
            #             'p22', 'p24', 'p25', 'p28', 'p4', 'p5R', 'p7L', 'p7R', 'p8', 'p15', 'pc']
        },
        "h_symmetry": [],
        "rotation_and_flips": {
            # "pne": ('v', 'h', 'd'),
            # "pn": ('v', 'h', 'd'),
            # "pnl": ('d',),
            # "pc": ('v', 'h', 'd'),
            # "pb": ('v', 'h', 'd'),
        }
    },
}
class_name = "RedRoundSign"


def get_siamese_model(base_model, image_shape=(224, 224, 3), merge_type='concatenate', add_batch_norm=False,
                      dropout=None):
    """
    create a VGG based, siamese network
    :param image_shape: shape of input ((224, 224, 3) if the original weight are used)
    :param weights: source of the weight to use on VGG, can by 'imagenet' or 'None' (or a path to weights)
    :param train_from_layers: number of layers to froze starting from the first layer (default 19: all)
    :param merge_type: how to merge the output of the siamese network, one of 'dot', 'multiply',
        'subtract', 'l1', 'l2' or 'concatenate'.
    :param add_batch_norm: True to add batch normalization before merging layers (default: False)
    :param layer_block_to_remove: How many block of layers to remove from VGG, starting at the end
        can be 0, 1, 2, 3 or 4, default 0.
    :param dropout: float: dropout to add in top model, None: no dropout (default: None)
    :return: a Keras model of a siamese VGG model with the given parameters
    """

    input_a = tf.keras.layers.Input(image_shape)
    input_b = tf.keras.layers.Input(image_shape)

    for layer in base_model.layers:
        layer.trainable = False

    siamese_vgg = get_siamese_layers(base_model, input_a, input_b,
                                     add_batch_norm=add_batch_norm,
                                     merge_type=merge_type)

    top = tf.keras.layers.Dense(256, activation="relu")(siamese_vgg)
    if dropout is not None:
        top = tf.keras.layers.Dropout(dropout)(top)
    top = tf.keras.layers.Dense(32, activation="relu")(top)
    if dropout is not None:
        top = tf.keras.layers.Dropout(dropout)(top)
    top = tf.keras.layers.Dense(2, activation="softmax")(top)

    return tf.keras.models.Model(inputs=[input_a, input_b], outputs=top)


def unfroze_core_model_layers(siamese_model: tf.keras.models.Model, number_of_layers_to_froze: int):
    """
    Get into the given siamese model to unfroze the last layers of the convectional part
    :param siamese_model: The siamese model which will be modified
    :param number_of_layers_to_froze: Number of layers where weight will not be trainable after this function,
    starting from last layers.
    :return: None
    """
    model_layer = siamese_model.layers[2]  # layers 0 and 1 are input, model is the third
    for layer in model_layer.layers[:number_of_layers_to_froze]:
        layer.trainable = False
    for layer in model_layer.layers[number_of_layers_to_froze:]:
        layer.trainable = True


def save_results(path: str, history: pd.DataFrame):
    """
    Save the given history to the given path with some plots of the results
    :param path: path were all the files will be created
    :param history: a panda dataframe containing the information about the learning process, mostly the information
    from model.fit.
    :return: None
    """
    if path is not None:
        os.makedirs(path, exist_ok=True)
        # write history to csv for late160r use

        history.to_csv(os.path.join(path, 'history.csv'), sep=',')
        # do some fancy plots
        # Accuracy
        plt.figure()
        plt.plot(history.get('epoch'), history.get('val_categorical_accuracy'), label='val_categorical_accuracy')
        plt.plot(history.get('epoch'), history.get('categorical_accuracy'), label='categorical_accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend()
        plt.savefig(os.path.join(path, 'accuracy.png'))
        # Loss
        plt.figure()
        plt.plot(history.get('epoch'), history.get('val_loss'), label='val_loss')
        plt.plot(history.get('epoch'), history.get('loss'), label='loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend()
        plt.savefig(os.path.join(path, 'loss.png'))
        # Evaluate model and save results


def main():
    # setup tensorflow backend (prevent "Blas SGEMM launch failed" error)
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    # config.log_device_placement = True  # to log device placement (on which device the operation ran)
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)  # set this TensorFlow session as the default session for Keras
    """
    Build and train a VGG Siamese network using the provided command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch-size',
                        default=64,
                        type=int,
                        dest="batch_size")
    parser.add_argument('-bn', '--batch-norm',
                        default=1,
                        type=int,
                        dest="use_batch_norm")
    parser.add_argument('-ml', '--merge-layer',
                        default='concatenate',
                        type=str,
                        dest="merge_layer")
    parser.add_argument('-o', '--out-dir',
                        default=None,
                        type=str,
                        dest="output_dir")
    parser.add_argument('-e', '--epochs-per-step',
                        default=1,
                        type=int,
                        dest="number_of_epoch")
    parser.add_argument('-ef', '--epochs-first-step',
                        default=None,
                        type=int,
                        dest="number_of_epoch_first_step")
    parser.add_argument('-d', '--dropout',
                        default=0.0,
                        type=float,
                        dest="dropout")
    parser.add_argument('-lr', '--learning-rate',
                        default=0.001,
                        type=float,
                        dest="learning_rate")
    parser.add_argument('-lrd', '--learning-rate-decay',
                        default=0.0005,
                        type=float,
                        dest="learning_rate_decay")
    parser.add_argument('-op', '--optimizer',
                        default='sgd',
                        type=str,
                        dest="optimizer")
    parser.add_argument('-f', '--fine-tuning-iteration',
                        default=0,
                        type=int,
                        dest="fine_tuning_iteration")
    parser.add_argument('-dp', '--dataset-path',
                        default="./data",
                        type=str,
                        dest="data_set_path")
    parser.add_argument('-m', '--model-name',
                        required=False,
                        default="MobileNetV2",
                        type=str,
                        dest="model_name")
    parser.add_argument('-ri', '--use-random-weight-initialisation',
                        required=False,
                        default=False,
                        type=bool,
                        dest="random_init")
    parser.add_argument('-ignore-npz', '--ignore-precomputed-learning-file',
                        required=False,
                        default=False,
                        type=bool,
                        dest="ignore_npz")
    parser.add_argument('-is', '--input-size',
                        required=False,
                        default=96,
                        type=int,
                        dest="input_size")
    parser.add_argument('-sp', '--same-proba',
                        required=False,
                        default=0.5,
                        type=float,
                        dest="same_proba")
    args = parser.parse_args()
    if args.output_dir is not None:
        model_save_path = args.output_dir + "_models"
        os.makedirs(model_save_path, exist_ok=True)
    else:
        model_save_path = None
    if args.number_of_epoch_first_step is None:
        args.number_of_epoch_first_step = args.number_of_epoch

    # check dataset
    if not os.path.isdir(args.data_set_path):
        raise ValueError("the specified dataset directory ('{}') is not a directory".format(args.data_set_path))

    out_classes = classes[class_name]["signs_classes"]
    rotation_and_flips = classes[class_name]["rotation_and_flips"]
    h_symmetry_classes = classes[class_name]["h_symmetry"]
    try:
        merge_sign_classes = classes[class_name]["merge_sign_classes"]
    except KeyError:
        merge_sign_classes = None
    os.makedirs(class_name, exist_ok=True)
    mapping = {c: i for i, c in enumerate(out_classes)}
    mapping_id_to_name = {i: c for c, i in mapping.items()}

    if args.random_init:
        weights = None
    else:
        weights = 'imagenet'
    if args.model_name == "MobileNetV2":
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
        base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(weights=weights,
                                                                    include_top=False,
                                                                    input_shape=(args.input_size,
                                                                                 args.input_size, 3),
                                                                    pooling='avg')
    elif args.model_name == "InceptionResNetV2":
        preprocess_input = tf.keras.applications.inception_resnet_v2.preprocess_input
        base_model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(weights=weights,
                                                                                 include_top=False,
                                                                                 input_shape=(args.input_size,
                                                                                              args.input_size, 3),
                                                                                 pooling='avg')
    elif args.model_name == "NASNetLarge":
        preprocess_input = tf.keras.applications.nasnet.preprocess_input
        base_model = tf.keras.applications.nasnet.NASNetLarge(weights=weights,
                                                              include_top=False,
                                                              input_shape=(args.input_size, args.input_size, 3),
                                                              pooling='avg')
    else:
        raise ValueError("unknown model name {}, should be one of {}".format(args.model_name,
                                                                             ["MobileNetV2", "InceptionResNetV2",
                                                                              "NASNetLarge"]))

    model = get_siamese_model(add_batch_norm=args.use_batch_norm,
                              merge_type=args.merge_layer,
                              dropout=args.dropout,
                              base_model=base_model,
                              image_shape=(args.input_size, args.input_size, 3))
    model.summary()

    model.compile(loss='categorical_crossentropy',
                  # categorical_crossentropy with 2 labels is the same than binary_crossentropy
                  optimizer=args.optimizer,
                  metrics=['categorical_accuracy'])

    x_train, y_train, x_val, y_val = get_data_for_master_class(class_name=class_name,
                                                               mapping=mapping,
                                                               mapping_id_to_name=mapping_id_to_name,
                                                               rotation_and_flips=rotation_and_flips,
                                                               data_dir=args.data_set_path,
                                                               merge_sign_classes=merge_sign_classes,
                                                               h_symmetry_classes=h_symmetry_classes,
                                                               image_size=(args.input_size, args.input_size),
                                                               ignore_npz=args.ignore_npz,
                                                               out_classes=out_classes)

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=[0.8, 1.1],
        fill_mode='nearest',
        horizontal_flip=False,
        vertical_flip=False,
        # validation_split=1.0 - train_ratio,
        brightness_range=(0.7, 1.3),
        preprocessing_function=preprocess_input
    )
    datagen_val = tf.keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=0,
        width_shift_range=0.0,
        height_shift_range=0.0,
        zoom_range=[1.0, 1.0],
        fill_mode='nearest',
        horizontal_flip=False,
        vertical_flip=False,
        brightness_range=[1.0, 1.0],
        preprocessing_function=preprocess_input
    )

    # datagen.fit(x_train)
    # datagen_val.mean = datagen.mean
    # datagen_val.std = datagen.std
    # datagen.standardize(x_train)
    # datagen_val.standardize(x_val)
    # with open(os.path.join(args.output_dir, "standardisation_param.json"), 'w') as j:
    #     json.dump()

    triple_sequence_train = TripleGenerator(x_train, y_train, generator=datagen, batch_size=args.batch_size,
                                            epoch_len=int(math.ceil(100000/args.batch_size)),
                                            same_proba=args.same_proba)
    triple_sequence_val = TripleGenerator(x_val, y_val, generator=datagen_val, batch_size=args.batch_size,
                                          epoch_len=int(math.ceil(10000/args.batch_size)), same_proba=0.5)

    print("fit done")

    # train the top layer of the classifier
    history = model.fit_generator(generator=triple_sequence_train,
                                  epochs=args.number_of_epoch_first_step,
                                  verbose=1,
                                  validation_data=triple_sequence_val,
                                  initial_epoch=0,
                                  use_multiprocessing=True
                                  )
    # save the result for analysis
    epoch = history.epoch
    h_values = history.history.values()
    values = np.array([epoch, ] + list(h_values) + [[0] * len(epoch)])
    df_history = pd.DataFrame(data=values.T,
                              columns=["epoch", ] + list(history.history.keys()) + ['fine_tuning']
                              )
    # save current model to disk if a path was specified
    if model_save_path is not None:
        model.save(os.path.join(model_save_path, "model0.h5"), overwrite=True)

    # update optimizer for fine tuning
    if args.optimizer == 'adam':
        opt = tf.keras.optimizers.Adam(lr=args.learning_rate,
                                       decay=args.learning_rate_decay)
    elif args.optimizer == 'rmsprop':
        opt = tf.keras.optimizers.RMSprop(lr=args.learning_rate,
                                          decay=args.learning_rate_decay)
    elif args.optimizer == 'sgd':
        opt = tf.keras.optimizers.SGD(lr=args.learning_rate,
                                      decay=args.learning_rate_decay)
    else:
        raise ValueError("Optimizer argument must be one of 'adam' or 'rmsprop', not " + str(args.optmizer))

    # now do some fine tuning if asked from command line
    for i in range(1, args.fine_tuning_iteration + 1):
        # unfroze the model a bit more
        unfroze_core_model_layers(model, 0)

        # compile the model to apply the modifications
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['categorical_accuracy'])

        # fit the model with this new freedom
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_save_path,
                                  "weights.{epoch:02d}-{val_loss:.2f}-{val_categorical_accuracy:.2f}.h5"),
            monitor="val_categorical_accuracy",
            save_weights_only=True,
            period=10
        )
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_categorical_accuracy", patience=10)
        initial_epoch = args.number_of_epoch_first_step + (i - 1) * args.number_of_epoch
        history = model.fit_generator(generator=triple_sequence_train,
                                      epochs=initial_epoch + args.number_of_epoch,
                                      verbose=1,
                                      validation_data=triple_sequence_val,
                                      initial_epoch=initial_epoch,
                                      use_multiprocessing=True,
                                      callbacks=[checkpoint, early_stopping]
                                      )
        # save the result for analysis
        epoch = history.epoch
        h_values = history.history.values()
        values = np.array([epoch, ] + list(h_values) + [[i] * len(epoch)])
        df_history = df_history.append(pd.DataFrame(data=values.T,
                                                    columns=["epoch"] + list(history.history.keys()) + ['fine_tuning']))
        # save current model to disk if a path was specified
        if model_save_path is not None:
            model.save(os.path.join(model_save_path, "model" + str(i) + ".h5"), overwrite=True)

    # add the training argument to history, to make filtering easier when all
    # the different history will be merged together.
    for k, v in args.__dict__.items():
        kwargs = {k: [v] * df_history.shape[0]}
        df_history = df_history.assign(**kwargs)

    save_results(args.output_dir, df_history)

    # # Evaluate training, just to be sure...
    # datagen_test = tf.keras.preprocessing.image.ImageDataGenerator(
    #     featurewise_center=False,
    #     featurewise_std_normalization=False,
    #     rotation_range=0,
    #     width_shift_range=0.0,
    #     height_shift_range=0.0,
    #     zoom_range=[1.0, 1.0],
    #     fill_mode='nearest',
    #     horizontal_flip=False,
    #     vertical_flip=False,
    #     brightness_range=[1.0, 1.0],
    #     preprocessing_function=preprocess_input
    # )
    # triple_sequence_test = TripleGenerator(x_val, y_val, generator=datagen_test, batch_size=args.batch_size,
    #                                        epoch_len=int(math.ceil(10000 / args.batch_size)), same_proba=0.5)
    #
    # model.evaluate(x=triple_sequence_test, verbose=1, use_multiprocessing=True)


if __name__ == '__main__':
    main()
