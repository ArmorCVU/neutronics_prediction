import random
import os

ASS_NUM = 177

C1_load_path = "/vol7/home/dep1/hualong/private/hualong3180_FM/3D_FM/C1_load/input/CORCA-3D.inp"

ass_pos = [[7,2],[8,2],[9,2],[10,2],[11,2],
           [5,3],[6,3],[7,3],[8,3],[9,3],[10,3],[11,3],[12,3],[13,3],
           [4,4],[5,4],[6,4],[7,4],[8,4],[9,4],[10,4],[11,4],[12,4],[13,4],[14,4],
           [3,5],[4,5],[5,5],[6,5],[7,5],[8,5],[9,5],[10,5],[11,5],[12,5],[13,5],[14,5],[15,5],
           [3,6],[4,6],[5,6],[6,6],[7,6],[8,6],[9,6],[10,6],[11,6],[12,6],[13,6],[14,6],[15,6],
           [2,7],[3,7],[4,7],[5,7],[6,7],[7,7],[8,7],[9,7],[10,7],[11,7],[12,7],[13,7],[14,7],[15,7],[16,7],
           [2,8],[3,8],[4,8],[5,8],[6,8],[7,8],[8,8],[9,8],[10,8],[11,8],[12,8],[13,8],[14,8],[15,8],[16,8],
           [2,9],[3,9],[4,9],[5,9],[6,9],[7,9],[8,9],[9,9],[10,9],[11,9],[12,9],[13,9],[14,9],[15,9],[16,9],
           [2,10],[3,10],[4,10],[5,10],[6,10],[7,10],[8,10],[9,10],[10,10],[11,10],[12,10],[13,10],[14,10],[15,10],[16,10],
           [2,11],[3,11],[4,11],[5,11],[6,11],[7,11],[8,11],[9,11],[10,11],[11,11],[12,11],[13,11],[14,11],[15,11],[16,11],
           [3,12],[4,12],[5,12],[6,12],[7,12],[8,12],[9,12],[10,12],[11,12],[12,12],[13,12],[14,12],[15,12],
           [3,13],[4,13],[5,13],[6,13],[7,13],[8,13],[9,13],[10,13],[11,13],[12,13],[13,13],[14,13],[15,13],
           [4,14],[5,14],[6,14],[7,14],[8,14],[9,14],[10,14],[11,14],[12,14],[13,14],[14,14],
           [5,15],[6,15],[7,15],[8,15],[9,15],[10,15],[11,15],[12,15],[13,15],
           [7,16],[8,16],[9,16],[10,16],[11,16]
           ]
ass_pos_new = [[7,2,-1],[8,2,-1],[9,2,-1],[10,2,-1],[11,2,-1],
           [5,3,-1],[6,3,-1],[7,3,-1],[8,3,-1],[9,3,-1],[10,3,-1],[11,3,-1],[12,3,-1],[13,3,-1],
           [4,4,-1],[5,4,-1],[6,4,-1],[7,4,-1],[8,4,-1],[9,4,-1],[10,4,-1],[11,4,-1],[12,4,-1],[13,4,-1],[14,4,-1],
           [3,5,-1],[4,5,-1],[5,5,-1],[6,5,-1],[7,5,-1],[8,5,-1],[9,5,-1],[10,5,-1],[11,5,-1],[12,5,-1],[13,5,-1],[14,5,-1],[15,5,-1],
           [3,6,-1],[4,6,-1],[5,6,-1],[6,6,-1],[7,6,-1],[8,6,-1],[9,6,-1],[10,6,-1],[11,6,-1],[12,6,-1],[13,6,-1],[14,6,-1],[15,6,-1],
           [2,7,-1],[3,7,-1],[4,7,-1],[5,7,-1],[6,7,-1],[7,7,-1],[8,7,-1],[9,7,-1],[10,7,-1],[11,7,-1],[12,7,-1],[13,7,-1],[14,7,-1],[15,7,-1],[16,7,-1],
           [2,8,-1],[3,8,-1],[4,8,-1],[5,8,-1],[6,8,-1],[7,8,-1],[8,8,-1],[9,8,-1],[10,8,-1],[11,8,-1],[12,8,-1],[13,8,-1],[14,8,-1],[15,8,-1],[16,8,-1],
           [2,9,-1],[3,9,-1],[4,9,-1],[5,9,-1],[6,9,-1],[7,9,-1],[8,9,-1],[9,9,-1],[10,9,-1],[11,9,-1],[12,9,-1],[13,9,-1],[14,9,-1],[15,9,-1],[16,9,-1],
           [2,10,-1],[3,10,-1],[4,10,-1],[5,10,-1],[6,10,-1],[7,10,-1],[8,10,-1],[9,10,-1],[10,10,-1],[11,10,-1],[12,10,-1],[13,10,-1],[14,10,-1],[15,10,-1],[16,10,-1],
           [2,11,-1],[3,11,-1],[4,11,-1],[5,11,-1],[6,11,-1],[7,11,-1],[8,11,-1],[9,11,-1],[10,11,-1],[11,11,-1],[12,11,-1],[13,11,-1],[14,11,-1],[15,11,-1],[16,11,-1],
           [3,12,-1],[4,12,-1],[5,12,-1],[6,12,-1],[7,12,-1],[8,12,-1],[9,12,-1],[10,12,-1],[11,12,-1],[12,12,-1],[13,12,-1],[14,12,-1],[15,12,-1],
           [3,13,-1],[4,13,-1],[5,13,-1],[6,13,-1],[7,13,-1],[8,13,-1],[9,13,-1],[10,13,-1],[11,13,-1],[12,13,-1],[13,13,-1],[14,13,-1],[15,13,-1],
           [4,14,-1],[5,14,-1],[6,14,-1],[7,14,-1],[8,14,-1],[9,14,-1],[10,14,-1],[11,14,-1],[12,14,-1],[13,14,-1],[14,14,-1],
           [5,15,-1],[6,15,-1],[7,15,-1],[8,15,-1],[9,15,-1],[10,15,-1],[11,15,-1],[12,15,-1],[13,15,-1],
           [7,16,-1],[8,16,-1],[9,16,-1],[10,16,-1],[11,16,-1]
           ]
ass_type = []
new_type_val = [18000, 24004, 24008, 31000, 31008, 31012, 31016, 39000, 44512, 44516, 44520, 49512]

def get_c1_load_ass_type():
    global ass_type
    lines = open(C1_load_path, "r").read()
    ass_type = lines.split("                TYPE : ")[1].split(";")[0].replace('"', '').replace(' ', '').replace('\n', '').split(",")

#切换坐标原点
def change_pos1(pos):
    return [pos[0] - 1 - 8, 8 - (pos[1] - 1)]


#切换坐标原点
def change_pos2(pos):
    return [pos[0] + 1 + 8, 8 - (pos[1] - 1)]

#1.1/8
#2.1/4+
#3.1/4x
#4.中心组件
def get_symmetry_type(pos):
    result_pos = change_pos1(pos)
    if result_pos[0] == 0 and result_pos[1] == 0:
        return 4
    elif abs(result_pos[0]) == abs(result_pos[1]):
        return 3
    elif abs(result_pos[0]) == 0 or abs(result_pos[1]) == 0:
        return 2
    else:
        return 1

def get_pos_num(pos):
    for i in range(len(ass_pos)):
        if ass_pos[i] == pos:
            return i
    

def add_pos_result_new(pos_result, pos, ass_flag, ass_type_val_num):
    pos_single = []
    pos_single.append(-1)
    pos_single.append(pos)
    pos_single.append(ass_type_val_num)
    pos_result[get_pos_num(pos)] = pos_single
    ass_flag[get_pos_num(pos)] = True

def generate_new_ass(ass_flag, pos_result):
    for i in range(len(ass_flag)):
        if ass_flag[i] == False:
            ass_type_val_num = random.randint(0, len(new_type_val) - 1)
            if get_symmetry_type(ass_pos[i]) == 4:
                pos_single = []
                pos_single.append(-1)
                pos_single.append(ass_pos[i])
                pos_single.append(ass_type_val_num)
                pos_result[i] = pos_single
                ass_flag[i] = True
            elif get_symmetry_type(ass_pos[i]) == 3:
                change_x = change_pos1(ass_pos[i])[0]
                change_y = change_pos1(ass_pos[i])[1]
                add_pos_result_new(pos_result, ass_pos[i], ass_flag, ass_type_val_num)
                add_pos_result_new(pos_result, change_pos2([-change_x,change_y]), ass_flag, ass_type_val_num)
                add_pos_result_new(pos_result, change_pos2([change_x,-change_y]), ass_flag, ass_type_val_num)
                add_pos_result_new(pos_result, change_pos2([-change_x,-change_y]), ass_flag, ass_type_val_num)
            elif get_symmetry_type(ass_pos[i]) == 2:
                change_x = change_pos1(ass_pos[i])[0]
                change_y = change_pos1(ass_pos[i])[1]
                add_pos_result_new(pos_result, ass_pos[i], ass_flag, ass_type_val_num)
                add_pos_result_new(pos_result, change_pos2([change_y,change_x]), ass_flag, ass_type_val_num)
                if change_x == 0:
                    add_pos_result_new(pos_result, change_pos2([change_x,-change_y]), ass_flag, ass_type_val_num)
                    add_pos_result_new(pos_result, change_pos2([-change_y,change_x]), ass_flag, ass_type_val_num)
                else:
                    add_pos_result_new(pos_result, change_pos2([-change_x,change_y]), ass_flag, ass_type_val_num)
                    add_pos_result_new(pos_result, change_pos2([change_y,-change_x]), ass_flag, ass_type_val_num)
            else:
                change_x = change_pos1(ass_pos[i])[0]
                change_y = change_pos1(ass_pos[i])[1]
                add_pos_result_new(pos_result, ass_pos[i], ass_flag, ass_type_val_num)
                add_pos_result_new(pos_result, change_pos2([-change_x,change_y]), ass_flag, ass_type_val_num)
                add_pos_result_new(pos_result, change_pos2([change_x,-change_y]), ass_flag, ass_type_val_num)
                add_pos_result_new(pos_result, change_pos2([-change_x,-change_y]), ass_flag, ass_type_val_num)

                add_pos_result_new(pos_result, change_pos2([change_y,change_x]), ass_flag, ass_type_val_num)
                add_pos_result_new(pos_result, change_pos2([-change_y,change_x]), ass_flag, ass_type_val_num)
                add_pos_result_new(pos_result, change_pos2([change_y,-change_x]), ass_flag, ass_type_val_num)
                add_pos_result_new(pos_result, change_pos2([-change_y,-change_x]), ass_flag, ass_type_val_num)

def add_pos_result_old(pos_result, old_pos, new_pos, ass_flag):
    pos_single = []
    pos_single.append(get_pos_num(old_pos))
    pos_single.append(new_pos)
    pos_result[get_pos_num(new_pos)] = pos_single
    ass_flag[get_pos_num(new_pos)] = True

#处理1/4x的情况
def generate_random3(temp_pos, pos_num, pos_result, ass_flag):
    old_ass_num = get_pos_num(temp_pos[pos_num])
    old_change_x = change_pos1(ass_pos[old_ass_num])[0]
    old_change_y = change_pos1(ass_pos[old_ass_num])[1]
    new_random_pos_num = 0 #可以移动的新坐标的个数
    new_ass_pos = [] #可以移动的新坐标的列表
    for i_pos in range(len(ass_pos)):
        x = change_pos1(ass_pos[i_pos])[0]
        y = change_pos1(ass_pos[i_pos])[1]
        if x != 0 and y != 0 and abs(x) == abs(y):
            new_random_pos_num += 1
            new_ass_pos.append(ass_pos[i_pos])
    new_random_num = random.randint(0, new_random_pos_num - 1) #随机一个坐标

    new_change_x = change_pos1(new_ass_pos[new_random_num])[0]
    new_change_y = change_pos1(new_ass_pos[new_random_num])[1]

    add_pos_result_old(pos_result, ass_pos[old_ass_num], new_ass_pos[new_random_num], ass_flag)
    add_pos_result_old(pos_result, change_pos2([-old_change_x, old_change_y]),
                        change_pos2([-new_change_y, new_change_x]), ass_flag)
    add_pos_result_old(pos_result, change_pos2([-old_change_x, -old_change_y]),
                        change_pos2([-new_change_x, -new_change_y]), ass_flag)
    add_pos_result_old(pos_result, change_pos2([old_change_x, -old_change_y]),
                        change_pos2([new_change_y, -new_change_x]), ass_flag)
    temp_pos.remove(ass_pos[old_ass_num])
    temp_pos.remove(change_pos2([-old_change_x, old_change_y]))
    temp_pos.remove(change_pos2([-old_change_x, -old_change_y]))
    temp_pos.remove(change_pos2([old_change_x, -old_change_y]))

#处理1/4+的情况
def generate_random2(temp_pos, pos_num, pos_result, ass_flag):
    old_ass_num = get_pos_num(temp_pos[pos_num])
    old_change_x = change_pos1(ass_pos[old_ass_num])[0]
    old_change_y = change_pos1(ass_pos[old_ass_num])[1]
    new_random_pos_num = 0 #可以移动的新坐标的个数
    new_ass_pos = [] #可以移动的新坐标的列表
    for i_pos in range(len(ass_pos)):
        x = change_pos1(ass_pos[i_pos])[0]
        y = change_pos1(ass_pos[i_pos])[1]
        if (x == 0 or y == 0) and not (x == 0 and y == 0):
            new_random_pos_num += 1
            new_ass_pos.append(ass_pos[i_pos])
    new_random_num = random.randint(0, new_random_pos_num - 1) #随机一个坐标

    new_change_x = change_pos1(new_ass_pos[new_random_num])[0]
    new_change_y = change_pos1(new_ass_pos[new_random_num])[1]
    add_pos_result_old(pos_result, ass_pos[old_ass_num], new_ass_pos[new_random_num], ass_flag)
    if old_change_x == 0:
        if new_change_x == 0:
            add_pos_result_old(pos_result, change_pos2([-old_change_y, old_change_x]),
                            change_pos2([-new_change_y, new_change_x]), ass_flag)
            add_pos_result_old(pos_result, change_pos2([old_change_y, old_change_x]),
                            change_pos2([new_change_y, new_change_x]), ass_flag)
            add_pos_result_old(pos_result, change_pos2([old_change_x, -old_change_y]),
                            change_pos2([new_change_x, -new_change_y]), ass_flag)

        else:
            add_pos_result_old(pos_result, change_pos2([-old_change_y, old_change_x]),
                            change_pos2([new_change_y, -new_change_x]), ass_flag)
            add_pos_result_old(pos_result, change_pos2([old_change_y, old_change_x]),
                            change_pos2([new_change_y, new_change_x]), ass_flag)
            add_pos_result_old(pos_result, change_pos2([old_change_x, -old_change_y]),
                            change_pos2([-new_change_x, new_change_y]), ass_flag)
        temp_pos.remove(change_pos2([-old_change_y, old_change_x]))
        temp_pos.remove(change_pos2([old_change_y, old_change_x]))
        temp_pos.remove(change_pos2([old_change_x, -old_change_y]))
    else:
        if new_change_y == 0:
            add_pos_result_old(pos_result, change_pos2([old_change_y, old_change_x]),
                            change_pos2([new_change_y, new_change_x]), ass_flag)
            add_pos_result_old(pos_result, change_pos2([-old_change_x, old_change_y]),
                            change_pos2([-new_change_x, new_change_y]), ass_flag)
            add_pos_result_old(pos_result, change_pos2([old_change_y, -old_change_x]),
                            change_pos2([new_change_y, -new_change_x]), ass_flag)
        else:
            add_pos_result_old(pos_result, change_pos2([old_change_y, old_change_x]),
                            change_pos2([new_change_y, new_change_x]), ass_flag)
            add_pos_result_old(pos_result, change_pos2([-old_change_x, old_change_y]),
                            change_pos2([new_change_x, -new_change_y]), ass_flag)
            add_pos_result_old(pos_result, change_pos2([old_change_y, -old_change_x]),
                            change_pos2([-new_change_y, new_change_x]), ass_flag)
        temp_pos.remove(change_pos2([old_change_y, old_change_x]))
        temp_pos.remove(change_pos2([-old_change_x, old_change_y]))
        temp_pos.remove(change_pos2([old_change_y, -old_change_x]))
    temp_pos.remove(ass_pos[old_ass_num])

#处理1/8的情况
def generate_random1(temp_pos, pos_num, pos_result, ass_flag):
    old_ass_num = get_pos_num(temp_pos[pos_num])
    old_change_x = change_pos1(ass_pos[old_ass_num])[0]
    old_change_y = change_pos1(ass_pos[old_ass_num])[1]
    new_random_pos_num = 0 #可以移动的新坐标的个数
    new_ass_pos = [] #可以移动的新坐标的列表

    for i_pos in range(len(ass_pos)):
        x = change_pos1(ass_pos[i_pos])[0]
        y = change_pos1(ass_pos[i_pos])[1]
        if abs(x) != abs(y) and x != 0 and y != 0:
            new_random_pos_num += 1
            new_ass_pos.append(ass_pos[i_pos])
    new_random_num = random.randint(0, new_random_pos_num - 1) #随机一个坐标

    new_change_x = change_pos1(new_ass_pos[new_random_num])[0]
    new_change_y = change_pos1(new_ass_pos[new_random_num])[1]

    add_pos_result_old(pos_result, ass_pos[old_ass_num], new_ass_pos[new_random_num], ass_flag)

    add_pos_result_old(pos_result, change_pos2([-old_change_y, old_change_x]),
                        change_pos2([-new_change_y, new_change_x]), ass_flag)
    add_pos_result_old(pos_result, change_pos2([old_change_x, -old_change_y]),
                        change_pos2([new_change_x, -new_change_y]), ass_flag)
    add_pos_result_old(pos_result, change_pos2([old_change_y, -old_change_x]),
                        change_pos2([new_change_y, -new_change_x]), ass_flag)

    add_pos_result_old(pos_result, change_pos2([-old_change_x, old_change_y]),
                        change_pos2([-new_change_x, new_change_y]), ass_flag)
    add_pos_result_old(pos_result, change_pos2([old_change_y, old_change_x]),
                        change_pos2([new_change_y, new_change_x]), ass_flag)
    add_pos_result_old(pos_result, change_pos2([-old_change_x, -old_change_y]),
                        change_pos2([-new_change_x, -new_change_y]), ass_flag)
    add_pos_result_old(pos_result, change_pos2([-old_change_y, -old_change_x]),
                        change_pos2([-new_change_y, -new_change_x]), ass_flag)

def generate_random():
    global ass_pos_new
    #装载数
    for num in range(3000):
        pos_result = [[] for _ in range(ASS_NUM)]
        composition_result = []
        #1.随机旧组件的个数
        old_ass_num = random.randint(0, 29)
        # old_ass_num = 5
        # print("old_ass_num:" + str(old_ass_num))
        #2.随机旧组件的坐标
        temp_pos = ass_pos.copy()
        ass_flag = [False for _ in range(ASS_NUM)] #判断该坐标是否已经存在组件
        for i in range(old_ass_num):
            pos_num = random.randint(0, len(temp_pos) - 1)
            # print("pos:" + str(temp_pos[pos_num][0]) + "," + str(temp_pos[pos_num][1]))
            if get_symmetry_type(temp_pos[pos_num]) == 4:
                old_ass_num = get_pos_num(temp_pos[pos_num])
                pos_single = []
                pos_single.append(old_ass_num)
                pos_single.append(change_pos2([0,0]))
                ass_flag[old_ass_num] = True
                new_ass_num = old_ass_num
                pos_result[new_ass_num] = pos_single
                temp_pos.remove([9, 9])
            elif get_symmetry_type(temp_pos[pos_num]) == 3:
                generate_random3(temp_pos, pos_num, pos_result, ass_flag)

            elif get_symmetry_type(temp_pos[pos_num]) == 2:
                generate_random2(temp_pos, pos_num, pos_result, ass_flag)

            else:
            	generate_random1(temp_pos, pos_num, pos_result, ass_flag)

        #3.随机新组件的坐标
        generate_new_ass(ass_flag, pos_result)

        #4.整理结果
        input_string = open("./input.dat", "r").read()
        if os.path.exists("./output_%s" % (num)):
            os.system("rm -r ./output_%s" % (num))
        os.system("mkdir ./output_%s" % (num))
        result_file = open("./output_%s/CORCA-3D.inp" % (num), "w")
        replace_string = "" #位置修改
        replace_string2 = "" #组件类型修改
        #确定新组件的名称
        for i_result in range(ASS_NUM):
            if pos_result[i_result][0] == -1:
                for i_ass_pos_new in range(ASS_NUM):
                    if pos_result[i_result][1][0] == ass_pos_new[i_ass_pos_new][0] and pos_result[i_result][1][1] == ass_pos_new[i_ass_pos_new][1]:
                        ass_pos_new[i_ass_pos_new][2] = 0
        ass_pos_new_num = 1
        old_num = 0 #修改了组件类型的个数
        for i_ass_pos_new in range(ASS_NUM):
            if ass_pos_new[i_ass_pos_new][2] == 0:
                ass_pos_new[i_ass_pos_new][2] = ass_pos_new_num
                ass_pos_new_num += 1

        for i_result in range(ASS_NUM):
            replace_string += "            @ASSEMBLY_BEG_%d\n" % (i_result + 1)
            if pos_result[i_result][0] != -1:
                replace_string += "            TYPE : " + '"%s";\n' % (ass_type[pos_result[i_result][0]])
                replace_string += '            FROM WHICH CYCLE : "CYCLE1";\n'
                replace_string += "            NAME : " + '"A%d";\n' %  (pos_result[i_result][0] + 1)
                replace_string += "            COORDINATE(X & Y) IN CORE INCLUDE REFLECTOR(OLD) : " + '%d,%d;\n' % (ass_pos[pos_result[i_result][0]][0], ass_pos[pos_result[i_result][0]][1])

                i_pos = pos_result[i_result][0]
                if ass_type[i_pos][-2:] != "00":
                    new_ass_name2 = ass_type[i_pos][:3] + "00"
                    new_ass_flag = False #判断新组件是否存在
                    for i_type in range(len(new_type_val)):
                        if new_ass_name2 == new_type_val[i_type]:
                            new_ass_flag = True
                    if new_ass_flag == True:
                        replace_string2 += "        @CHANGE_COMPOSITION_BEG_%d\n" % (old_num + 1)
                        replace_string2 += '            TYPE : ' + '"%s";\n' % (ass_type[pos_result[i_result][0]])
                        replace_string2 += "            NAME : " + '"A%d";\n' %  (pos_result[i_result][0] + 1)
                        replace_string2 += '            OLD COMPOSITON NAME : ' + '"%s";\n' % (ass_type[pos_result[i_result][0]])
                        replace_string2 += '            NEW COMPOSITON NAME : ' + '"%s";\n' % (ass_type[i_pos][:3] + "00")
                        replace_string2 += "        @CHANGE_COMPOSITION_END_%d\n" % (old_num + 1)
                        replace_string2 += '\n'
                        old_num += 1
            else:
                replace_string += "            TYPE : " + '"%s";\n' % (new_type_val[pos_result[i_result][2]])
                replace_string += '            FROM WHICH CYCLE : "NEW";\n'
                B = 0
                for i_ass_pos_new in range(ASS_NUM):
                    if pos_result[i_result][1][0] == ass_pos_new[i_ass_pos_new][0] and pos_result[i_result][1][1] == ass_pos_new[i_ass_pos_new][1]:
                        B = ass_pos_new[i_ass_pos_new][2]
                replace_string += "            NAME : " + '"B%d";\n' % (B)
                replace_string += "            COORDINATE(X & Y) IN CORE INCLUDE REFLECTOR(OLD) : -1,-1;\n"
            replace_string += "            COORDINATE(X & Y) IN CORE INCLUDE REFLECTOR(NEW) : " + '%d,%d;\n' % (pos_result[i_result][1][0], pos_result[i_result][1][1])
            replace_string += "            @ASSEMBLY_END_%d\n" % (i_result + 1)
            replace_string += "\n"
        replace_string2 = '        NUMBER OF ASSEMBLIES WHOSE COMPOSITION TO BE CHANGED : %d;\n' % (old_num) + replace_string2
        result = input_string.replace('replace1', replace_string).replace('replace2',replace_string2)
        result_file.write(result)
        result_file.close()

if __name__ == '__main__':
    get_c1_load_ass_type()
    generate_random()
