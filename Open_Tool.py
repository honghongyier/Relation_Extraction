#coding:utf-8
from numpy import random
import numpy as np
from gensim.models import word2vec
import re
from Relation import Relation
import os


class Open_tool:
    def get_pfmartrix(self, dim, size, path):
        '''
        # generating position matrix randomly
        # dim ==> column dimension of matrix
        # size ==> row dimension of matrix
        # path ==> path to save matrix
        '''
        right = size/2
        left = 1 - right
        pf_r1 = random.random(size=(size, dim))
        pf_r2 = random.random(size=(size, dim))
        if(path != ""):
            np.save(path + "/PF_R1.npy", pf_r1)
            np.save(path + "/PF_R2.npy", pf_r2)
        return pf_r1, pf_r2

    def load_word2vec(self, path, is_binary):
        # load your word2vec model
        vector = word2vec.Word2Vec.load_word2vec_format(path, binary=is_binary)
        return vector

    def sentence2vec(self, vector, v_dim, f_path, out_dic, pf_r1, pf_r2, pf_dim, pf_size, language):
        R = []
        type_cnt = 0
        type = {}
        names = os.listdir(f_path)
        for name in names:
            if language+"_" not in name:
                print("error"+name)
                continue
            f_in = open(f_path + '/' +name)
            print(name)
            # read file, get the word and value of relation position(PF)
            for line in f_in:
                line = line.replace("\n", '')
                R_temp = Relation()
                line = line.split('\t')
                if len(line) != 3:
                    print "Spliterror", name, line
                    continue
                R_temp.type = line[1]
                R_temp.sub_type = line[2]
                e1_raw_pos = []
                e2_raw_pos = []

                try:
                    arg1_temp = re.findall("<e1>(.*?)</e1>", line[0], re.S)[0]
                    arg1_temp = arg1_temp.replace("<e1>", "")
                    arg1_temp = arg1_temp.replace("</e1>", "")
                    R_temp.arg1 = arg1_temp
                    arg2_temp = re.findall("<e2>(.*?)</e2>", line[0], re.S)[0]
                    arg2_temp = arg2_temp.replace("<e2>", "")
                    arg2_temp = arg2_temp.replace("</e2>", "")
                    R_temp.arg2 = arg2_temp

                    e1_raw_pos.append(line[0].index("<e1>"))
                    e1_raw_pos.append(line[0].index("</e1>"))
                    e2_raw_pos.append(line[0].index("<e2>"))
                    e2_raw_pos.append(line[0].index("</e2>"))
                except IndexError, e:
                    arg1_temp = re.findall("<e1>(.*?)</e1", line[0], re.S)[0]
                    arg1_temp = arg1_temp.replace("<e1>", "")
                    arg1_temp = arg1_temp.replace("</e1", "")
                    R_temp.arg1 = arg1_temp
                    arg2_temp = re.findall("<e2>(.*?)</e2>", line[0], re.S)[0]
                    arg2_temp = arg2_temp.replace("<e2>", "")
                    arg2_temp = arg2_temp.replace("</e2>>", "")
                    R_temp.arg2 = arg2_temp

                    e1_raw_pos.append(line[0].index("<e1>"))
                    e1_raw_pos.append(line[0].index("</e1"))
                    e2_raw_pos.append(line[0].index("<e2>"))
                    e2_raw_pos.append(line[0].index("</e2>"))
                    print "Indexerror", name, line

                line_temp = line[0].split(' ')

                for i in range(0, len(line_temp)):
                    R_temp.PF['1'].append(i)
                    R_temp.PF['2'].append(i)
                    if "<e1>" in line_temp[i]:
                        R_temp.arg1_pos[0] = i
                        line_temp[i] = line_temp[i].replace("<e1>", "")
                    if "</e1>" in line_temp[i]:
                        R_temp.arg1_pos[1] = i
                        line_temp[i] = line_temp[i].replace("</e1>", "")
                    if "<e2>" in line_temp[i]:
                        R_temp.arg2_pos[0] = i
                        line_temp[i] = line_temp[i].replace("<e2>", "")
                    if "</e2>" in line_temp[i]:
                        R_temp.arg2_pos[1] = i
                        line_temp[i] = line_temp[i].replace("</e2>", "")

                middle_len = abs(R_temp.arg1_pos[0]-R_temp.arg2_pos[0]) #distances between two arguments
                str_arg1_temp = line_temp[R_temp.arg1_pos[0]:R_temp.arg1_pos[1]]
                str_arg2_temp = line_temp[R_temp.arg2_pos[0]:R_temp.arg2_pos[1]]

                # exclude sentences whose  middle_len >15
                if middle_len > 15:
                    continue
                max_len = 49
                # trimming sentence whose length > 49
                if len(line_temp) > max_len:
                    middle_start = min(R_temp.arg2_pos[0], R_temp.arg1_pos[0])
                    middle_end = max(R_temp.arg1_pos[1], R_temp.arg2_pos[1])
                    middle_len_arg = middle_end - middle_start;
                    if middle_len_arg > max_len:
                        continue
                    after_start = middle_start - (max_len-middle_len_arg)/2
                    after_end = middle_end + (max_len-middle_len_arg)/2
                    if after_start >= 0 and after_end <= len(line_temp):
                        line_temp = line_temp[after_start:after_end]
                        R_temp.arg1_pos[0] -= after_start
                        R_temp.arg1_pos[1] -= after_start
                        R_temp.arg2_pos[0] -= after_start
                        R_temp.arg2_pos[1] -= after_start
                        if str_arg1_temp != line_temp[R_temp.arg1_pos[0]:R_temp.arg1_pos[1]] and str_arg2_temp != line_temp[R_temp.arg2_pos[0]:R_temp.arg2_pos[1]]:
                            print "error1", line[0]
                    elif after_end < len(line_temp):
                        line_temp = line_temp[0:max_len]
                    elif after_start >= 0:

                        line_temp = line_temp[after_start:len(line_temp)]
                        R_temp.arg1_pos[0] -= after_start
                        R_temp.arg1_pos[1] -= after_start
                        R_temp.arg2_pos[0] -= after_start
                        R_temp.arg2_pos[1] -= after_start

                        if str_arg1_temp != line_temp[R_temp.arg1_pos[0]:R_temp.arg1_pos[1]] and str_arg2_temp != line_temp[R_temp.arg2_pos[0]:R_temp.arg2_pos[1]]:
                            print "error2", str_arg1_temp, line_temp[R_temp.arg1_pos[0]:R_temp.arg1_pos[1]], str_arg2_temp, line_temp[R_temp.arg2_pos[0]:R_temp.arg2_pos[1]]
                            #continue
                    else:
                        print "lenth error", name, len(line_temp)
                        continue

                R_temp.mention = line_temp
                for i in range(0, len(line_temp)):
                    R_temp.PF['1'][i] = i - R_temp.arg1_pos[0]
                    R_temp.PF['2'][i] = i - R_temp.arg2_pos[0]
                R_temp.mention_clean()
                #R_temp.combine(vector, v_dim, pf_r1, pf_r2, pf_dim, pf_size)
                R_temp.split(vector, v_dim, pf_r1, pf_r2, pf_dim, pf_size)
                if R_temp.type not in type.keys():
                    if not os.path.exists(out_dic + R_temp.type + "_" + str(type_cnt)):
                        os.mkdir(out_dic + R_temp.type + "_" + str(type_cnt))
                        type[R_temp.type] = [0, 0]
                        type[R_temp.type][0] = type_cnt
                        print(R_temp.type, type_cnt)
                        type_cnt += 1
                type[R_temp.type][1] += 1

                R.append(R_temp)

        return R, type
