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
    for i in range(5):
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

if  __name__ == '__main__':
    a= -0.55
    b = dtb(a)
    print(b)
    c = bttd(b)
    print(c)
