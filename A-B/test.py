import lasagne
import tensorflow as tf
import numpy as np

A_w = tf.sign(tf.Variable(tf.truncated_normal(shape=[8, 8], mean=0, stddev=0.1)))

def main(argv=None):
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        a = session.run([A_w])
        print(a)
    print("successful")


if __name__ == '__main__':
    tf.app.run()