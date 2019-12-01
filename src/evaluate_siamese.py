import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import math

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
        "rotation_and_flips": {"pne": ('v', 'h', 'd'),
                               "pn": ('v', 'h', 'd'),
                               "pnl": ('d',),
                               "pc": ('v', 'h', 'd'),
                               "pb": ('v', 'h', 'd'),
                               }
    },
}
class_name = "RedRoundSign"


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
    parser.add_argument('-dp', '--dataset-path',
                        default="./data",
                        type=str,
                        dest="data_set_path")
    parser.add_argument('-m', '--model-path',
                        required=True,
                        type=str,
                        dest="model_path")
    parser.add_argument('-mn', '--model-name',
                        required=False,
                        default="MobileNetV2",
                        type=str,
                        dest="model_name")
    parser.add_argument('-ignore-npz', '--ignore-precomputed-learning-file',
                        required=False,
                        default=False,
                        type=bool,
                        dest="ignore_npz")
    args = parser.parse_args()
    # os.makedirs(args.output_dir, exist_ok=True)

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

    model = tf.keras.models.load_model(args.model_path)
    model.summary()

    model.compile(loss='categorical_crossentropy',
                  # categorical_crossentropy with 2 labels is the same than binary_crossentropy
                  optimizer='sgd',
                  metrics=['categorical_accuracy'])

    input_size = int(model.input_shape[0][2])

    x_train, y_train, x_val, y_val = get_data_for_master_class(class_name=class_name,
                                                               mapping=mapping,
                                                               mapping_id_to_name=mapping_id_to_name,
                                                               rotation_and_flips=rotation_and_flips,
                                                               data_dir=args.data_set_path,
                                                               merge_sign_classes=merge_sign_classes,
                                                               h_symmetry_classes=h_symmetry_classes,
                                                               image_size=(input_size, input_size),
                                                               ignore_npz=args.ignore_npz,
                                                               out_classes=out_classes)

    if args.model_name == "MobileNetV2":
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    elif args.model_name == "InceptionResNetV2":
        preprocess_input = tf.keras.applications.inception_resnet_v2.preprocess_input
    elif args.model_name == "NASNetLarge":
        preprocess_input = tf.keras.applications.nasnet.preprocess_input
    else:
        raise ValueError("unknown model name {}, should be one of {}".format(args.model_name,
                                                                             ["MobileNetV2", "InceptionResNetV2",
                                                                              "NASNetLarge"]))

    datagen_test = tf.keras.preprocessing.image.ImageDataGenerator(
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

    # set values from red sign data set
    # datagen_test.mean = np.array([103.59205, 75.46247, 90.49107])
    # datagen_test.std = np.array([59.49902, 55.064148, 57.496548])
    # datagen_test.standardize(x_val)
    # datagen_test.standardize(x_train)

    triple_sequence_val = TripleGenerator(x_val, y_val, generator=datagen_test, batch_size=args.batch_size,
                                          epoch_len=int(math.ceil(len(x_val) * 10 / args.batch_size)), same_proba=0.5)
    triple_sequence_train = TripleGenerator(x_train, y_train, generator=datagen_test, batch_size=args.batch_size,
                                            epoch_len=int(math.ceil(len(x_train) * 10 / args.batch_size)),
                                            same_proba=0.5)

    print("fit done")

    print("Evaluating on validation split")
    model.evaluate(x=triple_sequence_val, verbose=1, use_multiprocessing=True)
    print("Evaluating on train split")
    model.evaluate(x=triple_sequence_train, verbose=1, use_multiprocessing=True)


if __name__ == '__main__':
    main()
