# -*- coding: UTF-8 -*

#去中括号[]
if  __name__ == '__main__':
   with open('C:/Users/mao/Desktop/a01.txt', 'r') as f:
       fb = open("C:/Users/mao/Desktop/a02.txt", 'w+')
       fb1 = open("C:/Users/mao/Desktop/a03.txt", 'w+')
       lines = f.readlines()
       chardigit = '-0123456789.'
       for line in lines:
           line = line.strip('\n')
           sts = ''
           for ch in line:

               if ch in chardigit:
                   sts = sts + ch
               else:
                   if ch == ',':
                        sts = sts + ' '
           fb.write(sts)

       fb.seek(0, 0)
       lines01 = fb.readlines()
       sts01 = ''
       i = 1
       for line in lines01:
           for ch in line:
               if ch != ' ':
                   sts01 = sts01 + ch
               else:
                   sts01 = sts01 + ' '
                   i = i + 1
               if (i % 17 == 0):
                   sts01 = sts01 + '\n'
                   i = 1
           print(sts01)
           fb1.write(sts01)
           fb1.write('\n')

       fb.close()
       fb1.close()