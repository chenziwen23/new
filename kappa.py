# coding=utf-8
import csv
from sklearn.metrics import cohen_kappa_score

filename1 = r'/home/czw/文档/234_.csv'  # 找出的元素

with open(filename1, 'r') as f_2:
    # 之前找出那些比较了多次属性的元素后，寸的时候没有存好， 把列表当str存进去了，这里用来处理这个问题的。改回成一个列表。
    readers = csv.reader(f_2)
    temp = []
    LID = []
    for j in readers:
        temp.append(j)
    for m in range(len(temp)):
        if len(temp[m]) == 6:
            temp[m][2] = temp[m][2].lstrip("['")
            temp[m][2] = temp[m][2].rstrip("'")
            temp[m][3] = temp[m][3].lstrip(" '")
            temp[m][3] = temp[m][3].rstrip("']")
            temp[m][4] = temp[m][4].lstrip("['")
            temp[m][4] = temp[m][4].rstrip("'")
            temp[m][5] = temp[m][5].lstrip(" '")
            temp[m][5] = temp[m][5].rstrip("']")
            temp2 = [temp[m][2], temp[m][3]]
            temp3 = [temp[m][4], temp[m][5]]
            LID.append([temp[m][0], temp[m][1], temp2, temp3])

        elif len(temp[m]) == 8:
            temp[m][2] = temp[m][2].lstrip("['")
            temp[m][2] = temp[m][2].rstrip("'")
            temp[m][3] = temp[m][3].lstrip(" '")
            temp[m][3] = temp[m][3].rstrip("'")
            temp[m][4] = temp[m][4].lstrip(" '")
            temp[m][4] = temp[m][4].rstrip("']")
            temp[m][5] = temp[m][5].lstrip("['")
            temp[m][5] = temp[m][5].rstrip("'")
            temp[m][6] = temp[m][6].lstrip(" '")
            temp[m][6] = temp[m][6].rstrip("'")
            temp[m][7] = temp[m][7].lstrip(" '")
            temp[m][7] = temp[m][7].rstrip("']")
            temp2 = [temp[m][2], temp[m][3], temp[m][4]]
            temp3 = [temp[m][5], temp[m][6], temp[m][7]]
            LID.append([temp[m][0], temp[m][1], temp2, temp3])

        elif len(temp[m]) == 10:
            temp[m][2] = temp[m][2].lstrip("['")
            temp[m][2] = temp[m][2].rstrip("'")
            temp[m][3] = temp[m][3].lstrip(" '")
            temp[m][3] = temp[m][3].rstrip("'")
            temp[m][4] = temp[m][4].lstrip(" '")
            temp[m][4] = temp[m][4].rstrip("'")
            temp[m][5] = temp[m][5].lstrip(" '")
            temp[m][5] = temp[m][5].rstrip("']")
            temp[m][6] = temp[m][6].lstrip("['")
            temp[m][6] = temp[m][6].rstrip("'")
            temp[m][7] = temp[m][7].lstrip(" '")
            temp[m][7] = temp[m][7].rstrip("'")
            temp[m][8] = temp[m][8].lstrip(" '")
            temp[m][8] = temp[m][8].rstrip("'")
            temp[m][9] = temp[m][9].lstrip(" '")
            temp[m][9] = temp[m][9].rstrip("']")
            temp2 = [temp[m][2], temp[m][3], temp[m][4], temp[m][5]]
            temp3 = [temp[m][6], temp[m][7], temp[m][8], temp[m][9]]
            LID.append([temp[m][0], temp[m][1], temp2, temp3])


def ins(temp, up, x, y):
    x = []
    y = []
    for m in range(len(temp)):
        # 求科恩卡怕系数：k
        dict = {}   # 元素形式：[id1, id2, ['safety','boring'],['left','equal']] 后面两个列表是对应的
        # print len(temp[m][2])
        for i in range(len(temp[m][2])):
            dict[temp[m][2][i]] = i
        if up[0] in dict.keys() and up[1] in dict.keys():
            J = dict[up[0]]   # 找出对应属性的winner
            K = dict[up[1]]
            if temp[m][3][J] == 'equal' or temp[m][3][K] == 'equal':
                continue
            if temp[m][3][J] == 'left':
                flag = 1
                x.append(flag)
            elif temp[m][3][J] == 'right':
                flag = -1
                x.append(flag)
            if temp[m][3][K] == 'left':
                flag = 1
                y.append(flag)
            elif temp[m][3][K] == 'right':
                flag = -1
                y.append(flag)
    k = cohen_kappa_score(x, y)
    print up[0] + '-----' + up[1]
    print k

a, b = [], []
ins(LID, ['safety', 'beautiful'], a, b)
ins(LID, ['safety', 'boring'], a, b)
ins(LID, ['safety', 'wealthy'], a, b)
ins(LID, ['safety', 'depressing'], a, b)
ins(LID, ['safety', 'lively'], a, b)
ins(LID, ['beautiful', 'boring'], a, b)
ins(LID, ['beautiful', 'wealthy'], a, b)
ins(LID, ['beautiful', 'depressing'], a, b)
ins(LID, ['beautiful', 'lively'], a, b)
ins(LID, ['boring', 'wealthy'], a, b)
ins(LID, ['boring', 'depressing'], a, b)
ins(LID, ['boring', 'lively'], a, b)
ins(LID, ['wealthy', 'depressing'], a, b)
ins(LID, ['wealthy', 'lively'], a, b)
ins(LID, ['depressing', 'lively'], a, b)
