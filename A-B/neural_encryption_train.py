import tensorflow as tf
import numpy as np
import Net
import os

import matplotlib
# OSX fix
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns

plainTextLength = 8
keyLength = 8
N = plainTextLength / 2
batch = 4096
learningRate = 0.0008
TRAIN_STEP= 10000
iterations = 1


def get_random_block(N, batch):
    return 2 * np.random.randint(2, size=(batch, N)) - 1

def train():
    with tf.name_scope('input_variable'):
        plain = tf.placeholder(tf.float32, shape=[None, plainTextLength], name='plainText')
        key = tf.placeholder(tf.float32, shape=[None, keyLength], name='keyText')

    Zeros = tf.zeros_like(plain, dtype=tf.float32, name='zeroVector')

    #
    Alice_output, Bob_output = Net._build_Network(plain, key, plainTextLength, keyLength)
    reshape_Bob_output = tf.reshape(Bob_output, shape=[-1, plainTextLength])
    # Bob L1 loss
    with tf.name_scope('Bob_loss'):
        Bob_loss = tf.reduce_mean(tf.abs(reshape_Bob_output - plain))
    tf.summary.scalar('Bob_loss_value', Bob_loss)

    # error
    boolean_P = tf.greater(plain, Zeros)
    boolean_B = tf.greater_equal(reshape_Bob_output, Zeros)
    accuracy_B = tf.reduce_mean(tf.cast(tf.equal(boolean_B, boolean_P), dtype=tf.float32))
    Bob_bits_wrong = plainTextLength - accuracy_B * plainTextLength
    tf.summary.scalar('accuracy_B_value', accuracy_B)
    tf.summary.scalar('Bob_bits_wrong', Bob_bits_wrong)

    A_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Alice')
    B_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Bob')
    AB_vars = A_vars + B_vars

    Alice_Bob_optimizer = tf.train.AdamOptimizer(learningRate).minimize(Bob_loss, var_list=AB_vars)

    merged = tf.summary.merge_all()

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        bob01_bits_wrong = []
        bob_acc  = []
        train_writer = tf.summary.FileWriter('./adver_logs', session.graph)
        if not os.path.exists('adver_logs'):
            os.makedirs('adver_logs')

        for step in range(TRAIN_STEP):
            # train Bob
            print ('Training Alice and Bob, Epoch:', step + 1)
            feedDict = {plain: get_random_block(plainTextLength, batch),
                        key: get_random_block(keyLength, batch)}
            for index in range(iterations):
                _, Bob_error, Bob_accuracy, Bob_wrong_bits,summary = session.run(
                    [Alice_Bob_optimizer, Bob_loss, accuracy_B, Bob_bits_wrong, merged], feed_dict=feedDict)
            Bob_accuracy_bits = Bob_accuracy * plainTextLength
            bob01_bits_wrong.append(Bob_wrong_bits)
            bob_acc.append(Bob_accuracy_bits)
            res_a = session.run([Alice_output], feedDict)
            print(Bob_accuracy_bits)
            print(Bob_wrong_bits)
            train_writer.add_summary(summary, step)

        sns.set_style("darkgrid")
        plt.plot(bob01_bits_wrong)
        plt.legend(['bob'])
        plt.xlabel('Epoch')
        plt.ylabel('bits_wrong achieved')
        plt.savefig("Graphname.png")

        saver = tf.train.Saver()
        saver.save(session,'model/save_net.ckpt',global_step=TRAIN_STEP)

def main(argv=None):
    train()

if __name__ == '__main__':
    tf.app.run()
