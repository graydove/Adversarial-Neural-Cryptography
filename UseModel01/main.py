import tensorflow as  tf
import  numpy as np
import Net
import time
import json
import csv

checkpoint_file = tf.train.latest_checkpoint("C:/Users/mao/Desktop/model01/")
plainTextLength = 16
keyLength = 16
batch = 4096

def get_random_block01(N = 16, batch = 4096):
    a = 2 * np.random.randint(2, size=(batch, N)) - 1
    np.savetxt("results/msg.txt", a, fmt = '%d')
    return a

def get_random_block02(N = 16, batch = 4096):
    a = 2 * np.random.randint(2, size=(batch, N)) - 1
    np.savetxt("results/key.txt", a, fmt = '%d')
    return a

def train():
    plain = tf.placeholder(tf.float32, shape=[None, plainTextLength], name='plainText')
    key = tf.placeholder(tf.float32, shape=[None, keyLength], name='keyText')

    #
    Alice_output, Alice_output01, Bob_output, Eve_output = Net._build_Network(plain, key, plainTextLength, keyLength)

    saver = tf.train.Saver()
    with tf.Session() as session:
        saver.restore(session, checkpoint_file)
        #train Bob
        feedDict = {plain: get_random_block01(plainTextLength + 8, batch),key: get_random_block02(keyLength, batch)}
        res_a = session.run([Alice_output], feedDict)
        res_a01 = session.run([Alice_output01], feedDict)
        res_b = session.run([Bob_output], feedDict)
        res_e = session.run([Eve_output], feedDict)


        np.set_printoptions(threshold=100000000000)  # 全部输出

        with  open('results/alice_{}.txt'.format(time.time()),'w' ) as f:
            f.write(json.dumps(np.array(res_a).tolist()))
        with  open('results/alice01_{}.txt'.format(time.time()), 'w') as f:
            f.write(str(res_a01))
        with  open('results/bob_{}.txt'.format(time.time()), 'w') as f:
            f.write(json.dumps(np.array(res_b).tolist()))
        with  open('results/eve_{}.txt'.format(time.time()),'w' ) as f:
            f.write(json.dumps(np.array(res_e).tolist()))
        print(res_a01)
        #print(res_a02)
        #print(res_a03)
        print(type(res_a01))


        print('---------------------------------------------------------')


def main(argv=None):
    train()
    print("successful")

if __name__ == '__main__':
    tf.app.run()