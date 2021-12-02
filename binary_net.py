#!/usr/bin/env python
# coding: utf-8

import os
import time
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Conv1D, BatchNormalization, Dropout, concatenate, Reshape, LeakyReLU

import matplotlib
# OSX fix
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns


learningRate = 0.0008
epoches = 10000
batch = 4096
plainTextLength = 16
keyLength = 16
aliceOutputLength = 16
unitsLength = 32


def get_random_block(N, batch):
    block = 2 * np.random.randint(2, size=(batch, N)) - 1
    return block.astype(np.float32)


class TruncatedNormal(keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(TruncatedNormal, self).__init__()
        w_init = tf.random.truncated_normal(shape=[units, input_dim],
                                            mean=0,
                                            stddev=0.1)
        self.w = tf.Variable(initial_value=w_init,
                             dtype='float32', trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w)


class TruncatedNormalE(keras.layers.Layer):
    def __init__(self, units=16, input_dim=32):
        super(TruncatedNormalE, self).__init__()
        w_init = tf.random.truncated_normal(shape=[units, input_dim],
                                            mean=0,
                                            stddev=0.1)
        self.w = tf.Variable(initial_value=w_init,
                             dtype='float32', trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w)


def build_Alice(plainTextLength, keyLength):
    unitLength = plainTextLength+keyLength
    input_plain = Input(shape=(plainTextLength, ), name='plain_input')
    input_key = Input(shape=(keyLength, ), name='key_input')
    x1 = concatenate([input_plain, input_key], axis=1, )
    x2 = TruncatedNormal()(x1)
    x3 = Reshape((unitLength, 1, ))(x2)
    x4 = Conv1D(filters=2, strides=1, kernel_size=4,
                padding='same',
                use_bias=False,)(x3)
    print('Alice-x4-shape:',x4.shape)
    x5 = BatchNormalization()(x4)
    x6 = LeakyReLU()(x5)
    x7 = Conv1D(filters=4, strides=2, kernel_size=2,
                padding='same',
                use_bias=False,)(x6)
    print('Alice-x7-shape:',x7.shape)
    x8 = BatchNormalization()(x7)
    x9 = LeakyReLU()(x8)
    x10 = Conv1D(filters=4, strides=1, kernel_size=1,
                 padding='same',
                 use_bias=False,)(x9)
    print('Alice-x10-shape:',x10.shape)
    x11 = BatchNormalization()(x10)
    x12 = LeakyReLU()(x11)
    x13 = Conv1D(filters=1, strides=1, kernel_size=1,
                 activation='tanh', padding='same',
                 use_bias=False,)(x12)
    print('Alice-x13-shape:',x13.shape)
    output = Reshape((plainTextLength, ))(x13)
    print('Alice-output-shape:',output.shape)

    model = Model(inputs=[input_plain, input_key], outputs=[output])
    return model


def build_Bob(aliceOutputLength, keyLength):
    unitLength = aliceOutputLength + keyLength
    input_alice = Input(shape=(aliceOutputLength, ), name='alice_input')
    input_key = Input(shape=(keyLength, ), name='key_input')
    x1 = concatenate([input_alice, input_key], axis=1, )
    x2 = TruncatedNormal()(x1)
    x3 = Reshape((unitLength, 1, ))(x2)
    x4 = Conv1D(filters=2, strides=1, kernel_size=4,
                padding='same',
                use_bias=False, kernel_initializer='truncated_normal')(x3)
    x5 = BatchNormalization()(x4)
    x6 = LeakyReLU()(x5)
    x7 = Conv1D(filters=4, strides=2, kernel_size=2,
                padding='same',
                use_bias=False, kernel_initializer='truncated_normal')(x6)
    x8 = BatchNormalization()(x7)
    x9 = LeakyReLU()(x8)
    x10 = Conv1D(filters=4, strides=1, kernel_size=1,
                 padding='same',
                 use_bias=False, kernel_initializer='truncated_normal')(x9)
    x11 = BatchNormalization()(x10)
    x12 = LeakyReLU()(x11)
    x13 = Conv1D(filters=1, strides=1, kernel_size=1,
                 activation='tanh', padding='same',
                 use_bias=False, kernel_initializer='truncated_normal')(x12)

    output = Reshape((aliceOutputLength, ))(x13)
    model = Model(inputs=[input_alice, input_key], outputs=[output])
    return model


def build_Eve(aliceOutputLength):
    input_alice = Input(shape=(aliceOutputLength, ))
    x1 = TruncatedNormalE()(input_alice)
    x2 = TruncatedNormal()(x1)
    x3 = Reshape((32, 1,))(x1)
    x4 = Conv1D(filters=2, strides=1, kernel_size=4,
                padding='same',
                use_bias=False, )(x3)
    x5 = BatchNormalization()(x4)
    x6 = LeakyReLU()(x5)
    x7 = Conv1D(filters=4, strides=2, kernel_size=2,
                padding='same',
                use_bias=False, )(x6)
    x8 = BatchNormalization()(x7)
    x9 = LeakyReLU()(x8)
    x10 = Conv1D(filters=4, strides=1, kernel_size=1,
                 padding='same',
                 use_bias=False,)(x9)
    x11 = BatchNormalization()(x10)
    x12 = LeakyReLU()(x11)
    x13 = Conv1D(filters=1, strides=1, kernel_size=1,
                 activation='tanh', padding='same',
                 use_bias=False,)(x12)
    output = Reshape((16, ))(x13)
    model = Model(inputs=[input_alice], outputs=[output])
    return model


def EveLoss(plain, EveOutput):
    loss = tf.reduce_mean(tf.abs(EveOutput - plain))
    return loss


def AliceBobLoss(plain, BobLoss, EveLoss):
    AliceBobLoss = BobLoss + (1 - EveLoss) ** 2
    return AliceBobLoss


def BobLoss(plain, BobOutput):
    loss = tf.reduce_mean(tf.abs(BobOutput - plain))
    return loss


#Alice_Bob_optimizer = keras.optimizers.Adam(learningRate)
#Eve_optimizer = keras.optimizers.Adam(learningRate)

Alice_Bob_optimizer = tf.optimizers.Adam(learningRate)
Eve_optimizer = tf.optimizers.Adam(learningRate)

Alice = build_Alice(plainTextLength, keyLength)
Bob = build_Bob(aliceOutputLength, keyLength)
Eve = build_Eve(aliceOutputLength)


checkpoint_dir = './checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(Alice_optimizer=Alice_Bob_optimizer,
                                 Bob_optimizer=Alice_Bob_optimizer,
                                 Eve_optimizer=Eve_optimizer,
                                 Alice=Alice,
                                 Bob=Bob,
                                 Eve=Eve)


def transform(output):
    output = tf.where(output < 0, output, 1 * tf.ones_like(output))
    output = tf.where(output > 0, output, -1 * tf.ones_like(output))
    return output

def train_step(plain, key, epoch):
    with tf.GradientTape() as Alice_tape, tf.GradientTape() as Bob_tape, tf.GradientTape() as Eve_tape:
        AliceOutput = Alice([plain, key], training=True)
        AliceOutput = transform(AliceOutput)
        BobOutput = Bob([AliceOutput, key], training=True)
        EveOutput = Eve(AliceOutput, training=True)
        bobLoss = BobLoss(plain, BobOutput)
        eveLoss = EveLoss(plain, EveOutput)
        alicebobLoss = AliceBobLoss(plain, bobLoss, eveLoss)
        Zeros = tf.zeros_like(plain, dtype=tf.float32, name='zeroVector')

        boolean_P = tf.greater(plain, Zeros)
        boolean_B = tf.greater_equal(BobOutput, Zeros)
        boolean_E = tf.greater_equal(EveOutput, Zeros)
        accuracy_B = tf.reduce_mean(
            tf.cast(tf.equal(boolean_B, boolean_P), dtype=tf.float32))
        accuracy_E = tf.reduce_mean(
            tf.cast(tf.equal(boolean_E, boolean_P), dtype=tf.float32))
        Bob_bits_wrong = plainTextLength - accuracy_B * plainTextLength
        Eve_bits_wrong = plainTextLength - accuracy_E * plainTextLength

        print(Bob_bits_wrong)
        print(Eve_bits_wrong)

    #print('AliceOutput: ', AliceOutput)
    #print('BobOutput: ', BobOutput)
    #print('EveOutput: ', EveOutput)
    # print('\n')

    gradients_of_Alice = Alice_tape.gradient(alicebobLoss,
                                             Alice.trainable_variables)
    gradients_of_Bob = Bob_tape.gradient(alicebobLoss,
                                         Bob.trainable_variables)
    gradients_of_Eve = Eve_tape.gradient(eveLoss, Eve.trainable_variables)
    Alice_Bob_optimizer.apply_gradients(zip(gradients_of_Alice,
                                            Alice.trainable_variables))
    Alice_Bob_optimizer.apply_gradients(zip(gradients_of_Bob,
                                            Bob.trainable_variables))
    Eve_optimizer.apply_gradients(zip(gradients_of_Eve,
                                      Eve.trainable_variables))

    if (epoch + 1) % 50 == 0:
        print_str = 'epoch: {}, Alice Loss: {}, Bob Loss: {}, Eve Loss: {},' +\
                    ' Bob bits wrong: {}, Eve bits wrong: {}'
        tf.print(print_str.format(epoch, alicebobLoss,
                                  bobLoss, eveLoss,
                                  Bob_bits_wrong,
                                  Eve_bits_wrong))
    return Bob_bits_wrong,Eve_bits_wrong

# In[303]:


def train(epochs, batch):
    bob01_bits_wrong, eve01_bits_wrong = [], []
    for epoch in range(epochs):
        print("train Epoch", epoch)
        start = time.time()
        plain = get_random_block(plainTextLength, batch),
        key = get_random_block(keyLength, batch)

        Bob_bits_wrong, Eve_bits_wrong = train_step(plain, key, epoch)
        bob01_bits_wrong.append(Bob_bits_wrong)
        eve01_bits_wrong.append(Eve_bits_wrong)
        if (epoch + 1) % 500 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        end = time.time()

    sns.set_style("darkgrid")
    plt.plot(bob01_bits_wrong)
    plt.plot(eve01_bits_wrong)
    plt.legend(['bob', 'eve', 'bob_acc', 'eve_acc'])
    plt.xlabel('Epoch')
    plt.ylabel('bits_wrong achieved')
    plt.savefig("Graphname.png")


# In[304]:

if __name__ == "__main__":
    train(epoches, batch)


# In[ ]:
