# from keras.models import Model
# from keras.layers import Input, Flatten, Dense, Dropout, Lambda, Concatenate, BatchNormalization, Subtract,\
#     Multiply
# from keras.preprocessing.image import ImageDataGenerator
# from keras import backend as K
import tensorflow as tf
import random
import numpy as np


def get_siamese_model(src_model: tf.keras.models.Model, input_shape: tuple, add_batch_norm: bool = False,
                      merge_type: str = 'concatenate'):
    """
    Create a siamese model from the given parameters
    :param src_model: model used for the siamese part, for example if vgg is provided here this will build a siamese
        network where using vgg to transform the images
    :param input_shape: shape of the input of the given model
    :param add_batch_norm: if batch normalisation must be added around the merge layers
    :param merge_type: how to merge the output of the siamese network, one of 'dot', 'multiply',
        'subtract', 'l1', 'l2' or 'concatenate'.
    :return: the siamese model
    """
    input_a = tf.keras.layers.Input(shape=input_shape)
    input_b = tf.keras.layers.Input(shape=input_shape)

    siamese = get_siamese_layers(src_model, input_a, input_b, add_batch_norm, merge_type)

    model = tf.keras.models.Model([input_a, input_b], siamese)
    return model


def get_siamese_layers(src_model: tf.keras.models.Model, input_a, input_b, add_batch_norm=False,
                       merge_type='concatenate'):
    """
    Create a set of layers needed to construct a siamese model (siamese structure plus merging method)
    :param src_model:  model used for the siamese part, for example if vgg is provided here this will build a siamese
        network where using vgg to transform the images
    :param input_a: input layer for one side of the model
    :param input_b: input layer for the other side of the model
    :param add_batch_norm: if batch normalisation must be added around the merge layers
    :param merge_type: how to merge the output of the siamese network, one of 'dot', 'multiply',
        'subtract', 'l1', 'l2' or 'concatenate'.
    :return: output layer of siamese part
    """
    processed_a = src_model(input_a)
    processed_b = src_model(input_b)
    if add_batch_norm:
        processed_a = tf.keras.layers.BatchNormalization(name='processed_a_normalization')(processed_a)
        processed_b = tf.keras.layers.BatchNormalization(name='processed_b_normalization')(processed_b)

    if merge_type == 'concatenate':
        siamese = tf.keras.layers.Concatenate(name='concatenate_merge')([processed_a, processed_b])
        siamese = tf.keras.layers.Flatten(name='concatenate_merge_flatten')(siamese)
    elif merge_type == 'dot':
        siamese = tf.keras.layers.Multiply(name='dot_merge_multiply')([processed_a, processed_b])
        siamese = tf.keras.layers.Lambda(lambda x: K.sum(x, axis=(1, 2)), name='dot_merge_sum')(siamese)
    elif merge_type == 'subtract':
        siamese = tf.keras.layers.Subtract(name='subtract_merge')([processed_a, processed_b])
        siamese = tf.keras.layers.Flatten(name='subtract_merge_flatten')(siamese)
    elif merge_type == 'l1':
        siamese = tf.keras.layers.Subtract(name='l1_merge_subtract')([processed_a, processed_b])
        siamese = tf.keras.layers.Lambda(lambda x: K.abs(x), name='l1_merge_abs')(siamese)
        siamese = tf.keras.layers.Flatten(name='l1_merge_flatten')(siamese)
    elif merge_type == 'l2':
        siamese = tf.keras.layers.Subtract(name='l2_merge_subtract')([processed_a, processed_b])
        siamese = tf.keras.layers.Lambda(lambda x: K.pow(x, 2), name='l2_merge_square')(siamese)
        siamese = tf.keras.layers.Flatten(name='l2_merge_flatten')(siamese)
    elif merge_type == 'multiply':
        siamese = tf.keras.layers.Multiply(name='multiply_merge')([processed_a, processed_b])
        siamese = tf.keras.layers.Flatten(name='multiply_merge_flatten')(siamese)
    else:
        raise ValueError("merge_type value incorrect, was " + str(merge_type) + " and not one of 'concatenate', 'dot', "
                         "'subtract', 'l1', 'l2' or 'multiply'")

    if add_batch_norm:
        siamese = tf.keras.layers.BatchNormalization(name='merge_normalisation')(siamese)

    return siamese


class TripleGenerator(tf.keras.utils.Sequence):
    def __init__(self, images_array: np.ndarray, classes_array: np.ndarray,
                 generator: tf.keras.preprocessing.image.ImageDataGenerator, same_proba: float = 0.5,
                 epoch_len: int = 10000, batch_size: int = 64):
        assert classes_array.ndim == 1
        self.images = images_array
        self.classes_dict = {}
        for i, c in enumerate(classes_array):
            if c not in self.classes_dict:
                self.classes_dict[c] = []
            self.classes_dict[c].append(i)
        for c, l in self.classes_dict.items():
            self.classes_dict[c] = np.array(l, dtype=np.int)
        self.classes_names = np.array(list(self.classes_dict.keys()), dtype=np.int)
        self.classes_prob = np.array([len(a) for a in self.classes_dict.values()], dtype=np.float32)
        self.classes_prob /= self.classes_prob.sum()
        self.length = epoch_len
        self.batch = batch_size
        self.same_proba = same_proba
        self.generator = generator

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        images_a, images_b, labels = [], [], []
        for i in range(self.batch):
            if random.random() > self.same_proba:
                ca, cb = np.random.choice(self.classes_names, size=2, replace=False, p=self.classes_prob)
                ia = np.random.choice(self.classes_dict[ca], size=1)[0]
                ib = np.random.choice(self.classes_dict[cb], size=1)[0]
                labels.append((0, 1))
            else:
                c = np.random.choice(self.classes_names, size=1, p=self.classes_prob)[0]
                ia, ib = np.random.choice(self.classes_dict[c], size=2, replace=False)
                labels.append((1, 0))
            images_a.append(self.generator.random_transform(self.images[ia]))
            images_b.append(self.generator.random_transform(self.images[ib]))
        images_b = np.stack(images_b)
        images_a = np.stack(images_a)
        labels = np.array(labels, dtype=np.int)
        return (images_a, images_b), labels


