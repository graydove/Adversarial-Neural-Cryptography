import tensorflow as  tf
import  numpy as np
import Net
import time
import json
import csv

np.set_printoptions(suppress=True)

checkpoint_file = tf.train.latest_checkpoint("C:/Users/mao/Desktop/model/")
plainTextLength = 16
keyLength = 16
batch = 4096


def strToFloat(number):
    try:
        return float(number)
    except:
        return number

def dtb(num):
    integercom = 0
    #判断是否为浮点数
    if num < 0:
        #若为负数
        integercom = 1

    num = abs(num)
    #取整数部分
    integer = int(num)
    #取小数部分
    flo = num - integer
    #小数部分进制转换
    tem = flo
    tmpflo = []
    for i in range(10):
        tem *= 2
        tmpflo += str(int(tem))
        tem -= int(tem)
        if tem == 0:
            break
    flocom = tmpflo
    integercom = str(integercom)
    return integercom + ''.join(flocom)

def bttd(num):
    d = 0
    print(str(num)[0])
    if str(num)[0] == '0':
        for index, ch in enumerate(str(num)[1:], start = 1):
            d  = d + int(ch)*(2**(-index))
            print(int(ch))
    else:
        for index, ch in enumerate(str(num)[1:], start = 1):
            d  = d - int(ch)*(2**(-index))
            print(int(ch))
    return d

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
    crypto = tf.placeholder(tf.float32, shape=[None, keyLength], name='keyText')
    b = get_random_block02(keyLength, batch)
    #
    Alice_output, Alice_output01 = Net._build_Network(plain, key, plainTextLength, keyLength)
    Bob_output, Eve_output = Net._build_Network01(crypto, key, plainTextLength, keyLength)

    saver = tf.train.Saver()
    with tf.Session() as session:
        saver.restore(session, checkpoint_file)
        #train Bob
        feedDict = {plain: get_random_block01(plainTextLength, batch),key: b}

        res_a = session.run([Alice_output], feedDict)
        res_a01 = session.run([Alice_output01], feedDict)
        res_a02 = [[0] * plainTextLength for _ in range(batch)]
        res_a03 = [[0] * plainTextLength for _ in range(batch)]
        print(res_a02[0][1])

        for i in range(batch):
            for j in range (plainTextLength):
                res_a02[i][j] = dtb(res_a[0][i][j])
                res_a03[i][j] = bttd(res_a02[i][j])
                print(res_a03[i][j])
        feedDict01 = {crypto: res_a03,key: b}
        res_b = session.run([Bob_output], feedDict01)
        res_e = session.run([Eve_output], feedDict01)

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
        print(type(res_a01))

        print('---------------------------------------------------------')


def main(argv=None):
    train()
    print("successful")

if __name__ == '__main__':
    tf.app.run()