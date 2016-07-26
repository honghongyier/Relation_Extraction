#coding:utf-8
'''
This is a preprocess python file for ACE2005 dataset
You can download this dataset from LDC

As for the word2vec embedding, we use pre-trained word embedding
Which could download here: linking


'''
import numpy as np
from numpy import random
from gensim.models import word2vec
import re
from xml.dom import minidom
import os


class ACE_info:
    # go gain the relation mention from ACE dataset
    def __init__(self):
        self.mention = ""
        self.mention_pos = [0, 0]
        self.arg1 = ""
        self.arg1_pos = [0, 0]
        self.arg2 = ""
        self.arg2_pos = [0, 0]
        self.out_form = ""  # out as "...<e1>arg1</e1>...<e2>arg2</e2>"
        self.type = ""  # relation of this mention
        self.sub_type = ""  # sub relation of this mention

    def combine(self):
        # using mention, mention_pos, arg1, arg2, arg1_pos, arg2_pos to generate the out_form
        pos1 = [0, 0]
        pos2 = [0, 0]
        pos1[0] = self.arg1_pos[0] - self.mention_pos[0]
        pos1[1] = self.arg1_pos[1] - self.mention_pos[0]
        pos2[0] = self.arg2_pos[0] - self.mention_pos[0]
        pos2[1] = self.arg2_pos[1] - self.mention_pos[0]
        # add tag <e1>
        self.out_form = self.mention[0:pos1[0]] + "<e1>" + self.mention[pos1[0]:pos1[1]+1] + "</e1>" + self.mention[pos1[1]+1:len(self.mention)]

        # add tag <e2>
        if pos2[0] > pos1[1]:
            pos2[0] += 9
            pos2[1] += 9
        elif pos2[0] > pos1[0]:
            if pos2[1] <= pos1[1]:
                pos2[0] += 4
                pos2[1] += 4
            else:
                pos2[0] += 4
                pos2[1] += 9
        else:
            if pos2[1] > pos1[1]:
                pos2[1] += 9
            elif pos2[1] > pos1[0]:
                pos2[1] += 4
        if pos2[1] < pos2[0]:
            print("error")

        self.out_form = self.out_form[0:pos2[0]] + "<e2>" + self.out_form[pos2[0]:pos2[1]+1] + "</e2>" + self.out_form[pos2[1]+1:len(self.out_form)]
        self.out_form = self.out_form.replace('\t', '') + "\t" + self.type + "\t" + self.sub_type

    def show(self):
        print(self.out_form)


class ACE_neg:
    # to gain the 'Other' relation mention from ACE
    def __init__(self):
        self.mention_pos = []
        self.arg1_pos = []  # bias in raw_txt, as the type: [[1,2],[2,3]]
        self.arg2_pos = []
        self.raw_txt = ""   # raw doc from ACE
        self.file_name = ""
        self.txt = []   # bias for every sentence in raw_txt
        self.entity_pos = []    # bias in raw_txt, as the type: [[[2,3],[3,4]],[[]],[]]

    def show(self):
        print(self.mention_pos)
        print(self.arg1_pos)
        print(self.arg2_pos)
        print(self.entity_pos)

    def show_entity(self):
        for i in range(0, len(self.entity_pos)):
            for j in range(0, len(self.entity_pos[i])):
                print i, self.raw_txt[self.entity_pos[i][j][0]:self.entity_pos[i][j][1]+1]

    def get_raw_txt(self, path):
        file_in = open(path+self.file_name)
        flag = 1
        for line in file_in:
            if ">" in line and "</" not in line and flag == 1:
                line = re.findall('>(.*?)', line, re.S)[0]
                line += '\n'
                self.raw_txt += line.decode('utf-8')
            elif "</" in line and flag == 1:
                try:
                    line = re.findall('>(.*?)</.*?', line, re.S)[0]
                    line += '\n'
                    self.raw_txt += line.decode('utf-8')
                except IndexError, e:
                    line = re.findall('</.*?>(.*?)', line, re.S)[0]
                    line += '\n'
                    self.raw_txt += line
            elif "<" in line or flag == 0:
                if "<" in line:
                    line = re.findall('^(.*?)<', line, re.S)[0]
                    self.raw_txt += line
                    flag = 0
                elif "/>" in line:
                    flag = 1
                    line = re.findall('/>(.*?)', line, re.S)[0]
                    line += '\n'
                    self.raw_txt += line
            elif flag == 1:
                self.raw_txt += line
        #self.raw_txt = '\n' + self.raw_txt #针对en_bc

    def fenjv(self):
        # split raw_txt by the mark ".", "?", "!", "。", "？", "！"
        # extract each sentence from raw_txt and recording its offset
        J_pos = [j for j, i in enumerate(self.raw_txt) if i == '。' or i == '.']
        W_pos = [j for j, i in enumerate(self.raw_txt) if i == '？' or i == '?']
        G_pos = [j for j, i in enumerate(self.raw_txt) if i == '！' or i == '!']
        # gain the offset of split and sorted
        split_pos = []
        split_pos.extend(J_pos)
        split_pos.extend(W_pos)
        split_pos.extend(G_pos)
        split_pos = sorted(list(set(split_pos)))
        # add the offset for each sentence
        pos = [66, 66]
        for i in range(0,len(split_pos)):
            pos[1] = split_pos[i]
            temp = [0, 0]
            temp[0] = pos[0]
            temp[1] = pos[1]
            self.txt.append(temp)
            pos[0] = pos[1]+1

    def mention_has_entity(self, txt_pos, mention_pos):
        for i in range(len(mention_pos)):
            if mention_pos[i][0] >= txt_pos[0] and mention_pos[i][1] <= txt_pos[1]:
                return mention_pos[i]
        return [-1, -1]

    def mention_is_other(self, txt, entity_list):
        #已知mention，组合entity_list如果arg中没有相应的组合，则处理输出otrher_type
        mention_other = []
        for i in range(0, len(self.mention_pos)):
            for k in range(0, len(self.mention_pos[i])):
                # if the sentence is exists in mention
                if txt[0] <= self.mention_pos[i][k][0] and txt[1] >= self.mention_pos[i][k][1]:
                    for index_entity1 in range(0, len(entity_list)):
                        for index_entity2 in range(index_entity1+1, len(entity_list)):
                            # if index_entity1 and index_entity has already in arg1[] and arg2[], continue ;
                            # else add this mention into other type
                            #先判断这两个组合是否和arg1以及arg2重合
                            if (entity_list[index_entity1] == self.arg1_pos[i][k] and entity_list[index_entity2] == self.arg2_pos[i][k]) or (entity_list[index_entity1] == self.arg2_pos[i][k] and entity_list[index_entity2] == self.arg1_pos[i][k]):
                                continue    #相同则不处理
                            else:
                                mention_temp = self.raw_txt[txt[0]:txt[1]]
                                mention_temp2 = self.raw_txt[txt[0]:txt[1]]
                                e1_pos = [entity_list[index_entity1][0]-txt[0], entity_list[index_entity1][1]-txt[0]]
                                e2_pos = [entity_list[index_entity2][0]-txt[0], entity_list[index_entity2][1]-txt[0]]
                                mention_temp = mention_temp[0: e1_pos[0]] + "<e1>" + self.raw_txt[entity_list[index_entity1][0]:entity_list[index_entity1][1]+1] + "</e1>" \
                                           + mention_temp[e1_pos[1]+1: len(mention_temp)]
                                mention_temp2 = mention_temp2[0: e1_pos[0]] + "<e2>" + self.raw_txt[entity_list[index_entity1][0]:entity_list[index_entity1][1]+1] + "</e2>" \
                                           + mention_temp2[e1_pos[1]+1: len(mention_temp)]

                                #再添加<e2>
                                if e2_pos[0] > e1_pos[1]:
                                    e2_pos[0] += 9
                                    e2_pos[1] += 9
                                elif e2_pos[0] > e1_pos[0]:
                                    if e2_pos[1] <= e1_pos[1]:
                                        e2_pos[0] += 4
                                        e2_pos[1] += 4
                                    else:
                                        e2_pos[0] += 4
                                        e2_pos[1] += 9
                                else:
                                    if e2_pos[1] > e1_pos[1]:
                                        e2_pos[1] += 9
                                    elif e2_pos[1] > e1_pos[0]:
                                        e2_pos[1] += 4
                                if e2_pos[1] < e2_pos[0]:
                                    print "error"

                                mention_temp = mention_temp[0: e2_pos[0]] + "<e2>" + mention_temp[e2_pos[0]:e2_pos[1]+1] + "</e2>" \
                                            + mention_temp[e2_pos[1]+1: len(mention_temp2)]
                                mention_temp2 = mention_temp2[0: e2_pos[0]] + "<e1>" + mention_temp2[e2_pos[0]:e2_pos[1]+1] + "</e1>" \
                                            + mention_temp2[e2_pos[1]+1: len(mention_temp2)]

                                mention_other.append(mention_temp)
                                mention_other.append(mention_temp2)

                else:
                    for index_entity1 in range(0, len(entity_list)):
                        for index_entity2 in range(index_entity1+1, len(entity_list)):
                            mention_temp = self.raw_txt[txt[0]:txt[1]]
                            mention_temp2 = self.raw_txt[txt[0]:txt[1]]
                            e1_pos = [entity_list[index_entity1][0]-txt[0], entity_list[index_entity1][1]-txt[0]]
                            e2_pos = [entity_list[index_entity2][0]-txt[0], entity_list[index_entity2][1]-txt[0]]
                            #print(self.raw_txt[entity_list[index_entity1][0]:entity_list[index_entity1][1]+1].encode('utf-8'),
                            #      self.raw_txt[entity_list[index_entity2][0]:entity_list[index_entity2][1]+1].encode('utf-8'))
                            #先添加<e1>
                            mention_temp = mention_temp[0: e1_pos[0]] + "<e1>" + self.raw_txt[entity_list[index_entity1][0]:entity_list[index_entity1][1]+1] + "</e1>" \
                                           + mention_temp[e1_pos[1]+1: len(mention_temp)]
                            mention_temp2 = mention_temp2[0: e1_pos[0]] + "<e2>" + self.raw_txt[entity_list[index_entity1][0]:entity_list[index_entity1][1]+1] + "</e2>" \
                                           + mention_temp2[e1_pos[1]+1: len(mention_temp)]
                            #再添加<e2>
                            if e2_pos[0] > e1_pos[1]:
                                e2_pos[0] += 9
                                e2_pos[1] += 9
                            elif e2_pos[0] > e1_pos[0]:
                                if e2_pos[1] <= e1_pos[1]:
                                    e2_pos[0] += 4
                                    e2_pos[1] += 4
                                else:
                                    e2_pos[0] += 4
                                    e2_pos[1] += 9
                            else:
                                if e2_pos[1] > e1_pos[1]:
                                    e2_pos[1] += 9
                                elif e2_pos[1] > e1_pos[0]:
                                    e2_pos[1] += 4
                            if e2_pos[1] < e2_pos[0]:
                                print "error"


                            #mention_temp = mention_temp[0: e2_pos[0]] + "<e2>" + self.raw_txt[entity_list[index_entity2][0]:entity_list[index_entity2][1]+1] + "</e2>" \
                                          # + mention_temp[e2_pos[1]+1: len(mention_temp)]
                            #mention_temp2 = mention_temp2[0: e2_pos[0]] + "<e1>" + self.raw_txt[entity_list[index_entity2][0]:entity_list[index_entity2][1]+1] + "</e1>" \
                                           #+ mention_temp2[e2_pos[1]+1: len(mention_temp)]

                            mention_temp = mention_temp[0: e2_pos[0]] + "<e2>" + mention_temp[e2_pos[0]:e2_pos[1]+1] + "</e2>" \
                                        + mention_temp[e2_pos[1]+1: len(mention_temp2)]
                            mention_temp2 = mention_temp2[0: e2_pos[0]] + "<e1>" + mention_temp2[e2_pos[0]:e2_pos[1]+1] + "</e1>" \
                                        + mention_temp2[e2_pos[1]+1: len(mention_temp2)]
                            mention_other.append(mention_temp)
                            mention_other.append(mention_temp2)
        #print(mention_other)
        return list(set(mention_other))

    def get_other_type(self):
        # get the sentence which contains two mentions
        mention_list = []
        for i in range(0, len(self.txt)):
            entity_list = []
            for j in range(0, len(self.entity_pos)):
                temp_pos = self.mention_has_entity(self.txt[i], self.entity_pos[j])
                if temp_pos[0] != -1:
                    entity_list.append(temp_pos)
            if len(entity_list) >= 2:
                # gain the 'other' mention
                temp_list = self.mention_is_other(self.txt[i], entity_list)
                mention_list.extend(temp_list)
        return mention_list


class Relation:
    #Definition a class for turn relation mention to embedding
    def __init__(self):
        self.mention = ""   # content of mention
        self.mention_pos = [0, 0]   # bias of mention
        self.arg1 = ""  # content of arg1
        self.arg1_pos = [0, 0]  # bias of arg1
        self.arg2 = ""  # content of arg2
        self.arg2_pos = [0, 0]  # bias of arg2
        self.out_form = np.zeros((50, 400), dtype="float32")
        # embedding combined word-embedding and position embedding
        self.type = ""  # relation of this mention
        self.PF = {'1': [], '2': []}    # position embedding for each word
        self.sub_type = ""
        self.spilt_form = []
        # embedding combined with word-embedding and position embedding and
        # split into three parts for our CNN architecture

    def show(self):
        print(self.arg1, self.type, self.sub_type, self.arg2)
        print(self.mention)

    def mention_clean(self):
        # lower-casing the sentence and removing punctuation
        for i in range(0, len(self.mention)):
            self.mention[i] = self.mention[i].lower()
            for mark in [',', '.', '?']:
                self.mention[i] = self.mention[i].replace(mark, "")

    def combine(self, vector, v_dim, sen_dim, pf_r1, pf_r2, pf_dim, pf_size):
        '''
        # Combine the word-embedding and position embedding
        # vector ==> word2vec matrix
        # v_dim ==> dimension of word in vector
        # sen_dim ==> max length of sentence
        # pf_r1, pf_r2 ==> random matrix used to represent position embedding
        # pf_size ==> row_vector of pf_1 and pf_2
        '''
        for i_t in range(0, len(self.mention)):
            if i_t < sen_dim and self.mention[i_t] in vector:
                self.out_form[i_t, 0:v_dim] = vector[self.mention[i_t]]
                self.out_form[i_t, v_dim:v_dim+pf_dim] = pf_r1[int(self.PF['1'][i_t] + pf_size/2)]
                self.out_form[i_t, v_dim+pf_dim:len(self.out_form[i_t])] = pf_r2[int(self.PF['2'][i_t] + pf_size/2)]
            elif i_t < sen_dim:
                self.out_form[i_t, 0:v_dim] = 0
                self.out_form[i_t, v_dim:v_dim+pf_dim] = pf_r1[int(self.PF['1'][i_t] + pf_size/2)]
                self.out_form[i_t, v_dim+pf_dim:len(self.out_form[i_t])] = pf_r2[int(self.PF['2'][i_t] + pf_size/2)]
            elif i_t >= sen_dim:
                break
            else:
                self.out_form[i_t, 0:len(self.out_form[i_t])] = 0

    def split(self, vector, v_dim, pf_r1, pf_r2, pf_dim, pf_size):
        '''
        # Combine the word-embedding and position embedding
        # vector ==> word2vec matrix
        # v_dim ==> dimension of word in vector
        # sen_dim ==> max length of sentence
        # pf_r1, pf_r2 ==> random matrix used to represent position embedding
        # pf_size ==> row_vector of pf_1 and pf_2
        '''
        pos1 = min(self.arg1_pos[0], self.arg2_pos[0])
        pos2 = max(self.arg1_pos[0], self.arg2_pos[0])

        dim = 15
        temp_mat = np.zeros((dim, v_dim+pf_dim*2), dtype="float32")
        for i in range(0, pos1):
            if dim-1-i < 0:
                break
            if pos1-i >= 0 and self.mention[pos1-i] in vector:
                temp_mat[dim-1-i, 0:v_dim] = vector[self.mention[pos1-i]]
                temp_mat[dim-1-i, v_dim:v_dim+pf_dim] = pf_r1[int(self.PF['1'][pos1-i] + pf_size/2)]
                temp_mat[dim-1-i, v_dim+pf_dim:len(temp_mat[dim-1-i])] = pf_r2[int(self.PF['2'][pos1-i] + pf_size/2)]
            elif pos1-i >= 0:
                temp_mat[dim-1-i, 0:v_dim] = 0
                temp_mat[dim-1-i, v_dim:v_dim+pf_dim] = pf_r1[int(self.PF['1'][pos1-i] + pf_size/2)]
                temp_mat[dim-1-i, v_dim+pf_dim:len(temp_mat[dim-1-i])] = pf_r2[int(self.PF['2'][pos1-i] + pf_size/2)]
            else:
                temp_mat[dim-1-i, 0:len(temp_mat[dim-1-i])] = 0
        self.spilt_form.append(temp_mat)

        dim = 15
        temp_mat = np.zeros((dim, v_dim+pf_dim*2), dtype="float32")
        for i in range(pos1, pos2 + 1):
            if i-pos1 < dim and self.mention[i] in vector:
                temp_mat[i-pos1, 0:v_dim] = vector[self.mention[i]]
                temp_mat[i-pos1, v_dim:v_dim+pf_dim] = pf_r1[int(self.PF['1'][i] + pf_size/2)]
                temp_mat[i-pos1, v_dim+pf_dim:len(temp_mat[i-pos1])] = pf_r2[int(self.PF['2'][i] + pf_size/2)]
            elif i-pos1 < dim:
                temp_mat[i-pos1, 0:v_dim] = 0
                temp_mat[i-pos1, v_dim:v_dim+pf_dim] = pf_r1[int(self.PF['1'][i] + pf_size/2)]
                temp_mat[i-pos1, v_dim+pf_dim:len(temp_mat[i-pos1])] = pf_r2[int(self.PF['2'][i] + pf_size/2)]
            elif i-pos1 >= dim:
                break
            else:
                temp_mat[i-pos1, 0:len(temp_mat[i-pos1])] = 0
        self.spilt_form.append(temp_mat)

        dim = 20
        temp_mat = np.zeros((dim, v_dim+pf_dim*2), dtype="float32")
        for i in range(pos2 + 1, len(self.mention)):
            if i-pos2-1 < dim and self.mention[i] in vector:
                temp_mat[i-pos2-1, 0:v_dim] = vector[self.mention[i]]
                temp_mat[i-pos2-1, v_dim:v_dim+pf_dim] = pf_r1[int(self.PF['1'][i] + pf_size/2)]
                temp_mat[i-pos2-1, v_dim+pf_dim:len(temp_mat[i-pos2-1])] = pf_r2[int(self.PF['2'][i] + pf_size/2)]
            elif i-pos2-1 < dim:
                temp_mat[i-pos2-1, 0:v_dim] = 0
                temp_mat[i-pos2-1, v_dim:v_dim+pf_dim] = pf_r1[int(self.PF['1'][i] + pf_size/2)]
                temp_mat[i-pos2-1, v_dim+pf_dim:len(temp_mat[i-pos2-1])] = pf_r2[int(self.PF['2'][i] + pf_size/2)]
            elif i-pos2-1 >= dim:
                break
            else:
                temp_mat[i-pos2-1, 0:len(temp_mat[i-pos2-1])] = 0
        self.spilt_form.append(temp_mat)


def get_attrvalue(node, attrname):
    return node.getAttribute(attrname) if node else ''


def get_nodevalue(node, index = 0):
    return node.childNodes[index].nodeValue if node else ''


def get_xmlnode(node, name):
    return node.getElementsByTagName(name) if node else''


def extract_other(file_name):
    # parse xml from ACE data
    doc = minidom.parse(file_name)
    root = doc.documentElement
    R_element = ACE_neg()

    R_element.file_name = get_attrvalue(root, 'URI')
    relation_nodes = get_xmlnode(root, 'relation')
    entity_nodes = get_xmlnode(root, 'entity')

    for node in entity_nodes:
        entity_mention = get_xmlnode(node, 'entity_mention')
        temp_pos = []
        for mention_node in entity_mention:
            entity_extent = get_xmlnode(mention_node, 'charseq')
            pos = [0, 0]
            pos[0] = int(get_attrvalue(entity_extent[0], 'START'))
            pos[1] = int(get_attrvalue(entity_extent[0], 'END'))
            temp_pos.append(pos)
        R_element.entity_pos.append(temp_pos)

    for node in relation_nodes:
        mention_nodes = get_xmlnode(node, 'relation_mention')

        mention_pos = []
        arg1_pos = []
        arg2_pos = []

        for mention_node in mention_nodes:
            # gain the attribute info of mention
            mention_extent = get_xmlnode(mention_node, 'charseq')
            pos2 = [0, 0]
            pos2[0] = int(get_attrvalue(mention_extent[0], 'START'))
            pos2[1] = int(get_attrvalue(mention_extent[0], 'END'))
            mention_pos.append(pos2)

            argument_nodes = get_xmlnode(mention_node, 'relation_mention_argument')
            for argument_node in argument_nodes:
                argument_extent = get_xmlnode(argument_node, 'charseq')
                pos3 = [0, 0]
                pos3[0] = int(get_attrvalue(argument_extent[0], 'START'))
                pos3[1] = int(get_attrvalue(argument_extent[0], 'END'))
                if get_attrvalue(argument_node, 'ROLE') == 'Arg-1':
                    arg1_pos.append(pos3)
                elif get_attrvalue(argument_node, 'ROLE') == 'Arg-2':
                    arg2_pos.append(pos3)

        R_element.arg2_pos.append(arg2_pos)
        R_element.arg1_pos.append(arg1_pos)
        R_element.mention_pos.append(mention_pos)
    return R_element


def get_other(path_dic, path_out, language):
    f_out = open(path_out, 'w')
    f_names = os.listdir(path_dic)
    for name in f_names:
        if "apf.xml" not in name or 'score' in name:
            continue
        R = extract_other(path_dic + "/" + name)
        R.get_raw_txt(path_dic + "/")
        R.fenjv()
        other = R.get_other_type()
        if language == "ch":
            for i in range(0, len(other)):
                other[i] = other[i].replace('\t', ' ')
                f_out.write(other[i].replace('\n', ' ') + '\tother\tother\n')
        else:
            for i in range(0, len(other)):
                other[i] = other[i].replace('\t', ' ')
                f_out.write(other[i].replace('\n', ' ') + '\n')
    f_out.close()


def extract_info(file_name):
    R = []
    doc = minidom.parse(file_name)
    root = doc.documentElement

    relation_nodes = get_xmlnode(root, 'relation')
    relation_list = []
    for node in relation_nodes:
        relation_type = get_attrvalue(node, 'TYPE')
        relation_sub_type = get_attrvalue(node, 'SUBTYPE')
        mention_nodes = get_xmlnode(node, 'relation_mention')
        for mention_node in mention_nodes:
            R_element = ACE_info()
            R_element.type = relation_type
            R_element.sub_type = relation_sub_type

            # gain the attribute info of mention
            mention_extent = get_xmlnode(mention_node, 'charseq')
            R_element.mention_pos[0] = int(get_attrvalue(mention_extent[0], 'START'))
            R_element.mention_pos[1] = int(get_attrvalue(mention_extent[0], 'END'))
            R_element.mention = get_nodevalue(mention_extent[0])

            argument_nodes = get_xmlnode(mention_node, 'relation_mention_argument')
            for argument_node in argument_nodes:
                if get_attrvalue(argument_node, 'ROLE') == 'Arg-1':
                    # gain the attribute info of arg1
                    R_element.arg1_pos[0] = int(get_attrvalue(get_xmlnode(argument_node, 'charseq')[0], 'START'))
                    R_element.arg1_pos[1] = int(get_attrvalue(get_xmlnode(argument_node, 'charseq')[0], 'END'))
                    #relation_arg1 = get_nodevalue(get_xmlnode(argument_node, 'charseq')[0]).encode('utf-8', 'ignore')
                    #print(R_element.arg1_pos[0],R_element.arg1_pos[1])
                elif get_attrvalue(argument_node, 'ROLE') == 'Arg-2':
                    # gain the attribute info of age2
                    R_element.arg2_pos[0] = int(get_attrvalue(get_xmlnode(argument_node, 'charseq')[0], 'START'))
                    R_element.arg2_pos[1] = int(get_attrvalue(get_xmlnode(argument_node, 'charseq')[0], 'END'))
                    #relation_arg2 = get_nodevalue(get_xmlnode(argument_node, 'charseq')[0]).encode('utf-8', 'ignore')
                    #print(R_element.arg2_pos[0],R_element.arg2_pos[1])
            R_element.combine()
            R.append(R_element)
    return R


def get_info(path_dic, path_out):
    f_out = open(path_out, 'w')
    f_names = os.listdir(path_dic)
    for name in f_names:
        if "apf.xml" not in name or 'score' in name:
            continue
        R = extract_info(path_dic +"/" + name)
        for i in range(0, len(R)):
            R[i].out_form = R[i].out_form.replace('\n', ' ')
            f_out.write(R[i].out_form.encode('utf-8'))
            file_out.write('\n')
    f_out.close()


def get_pfmartrix(dim, size, path):
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


def load_word2vec(path, is_binary):
    # load your word2vec model
    vector = word2vec.Word2Vec.load_word2vec_format(path, is_binary)
    return vector


def sentence2vec(vector, v_dim, f_path, out_dic, pf_r1, pf_r2, pf_dim, pf_size, language):
    R = []
    type_cnt = 0
    type = {}

    for name in f_path:
        if language+"_" not in name:
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
            #R_temp.combine(vector, v_dim, 50, pf_r1, pf_r2, pf_dim, pf_size)
            R_temp.split(vector, v_dim, 50,pf_r1, pf_r2, pf_dim, pf_size)
            if R_temp.type not in type.keys():
                if not os.path.exists(out_dic + R_temp.type + "_" + str(type_cnt)):
                    os.mkdir(out_dic + R_temp.type + "_" + str(type_cnt))
                    type[R_temp.type] = [0, 0]
                    type[R_temp.type][0] = type_cnt
                    print(R_temp.type, type_cnt)
                    type_cnt += 1
            type[R_temp.type][1] += 1

            R.append(R_temp)

    return R,type


if __name__ == "__main__":

    # get 'other' mention and relation mention from ACE2005
    other_save_path = "out/ace2005/other_en_wl.txt"
    mention_save_path = "out/ace2005/en_wl.txt"
    directory = "Raw_Data/ace_2005_td_v7/data/English/wl/adj"
    get_other(directory, other_save_path, "en")  # you can change en to "ch" if the input in a chinese data set
    get_info(directory, mention_save_path)

    # change the mention into embedding

    # load word2vec
    vector_path = "out/GoogleNews-vectors-negative300.bin"
    vector_dim = 300
    vector = load_word2vec(vector_path, True)

    # load your position embedding matrix or generate a new one
    PF_dim = 50
    PF_size = 100
    PF_R1 = np.load("out/PF_R1_50.npy")
    PF_R2 = np.load("out/PF_R2_50.npy")

    '''
    PF_R1, PF_R2 = get_pfmartrix(PF_dim, PF_size, "out/") #generate a random embedding matrix
    '''


    # preprocessing start
    f_path = "out/ace2005"  # mention path
    out_dic = "out/ace2005/Data300_PF50/"
    R, type = sentence2vec(vector, vector_dim, f_path, out_dic, PF_R1, PF_R2, PF_dim, PF_size, "en")

    # save embeddings
    print "starting saving..."
    for key in type:
        print key
        #fea_mat = np.zeros((type[key][1], 50, vector_dim+PF_dim*2), dtype="float32")
        fea_mat_M = []
        cnt = 0
        for i in range(0, len(R)):
            if R[i].type == key:
                #fea_mat[cnt] = R[i].out_form
                fea_mat_M.append(R[i].spilt_form)
                cnt += 1
        #index_tr = [i for i in range(len(fea_mat))]
        #random.shuffle(index_tr)
        #fea_mat = [fea_mat[k] for k in index_tr]
        #np.save("out/ace2005/Data300_PF50/" + key + "_" + str(type[key][0]), fea_mat)
        np.save(out_dic+"M_" + key + "_" + str(type[key][0]), fea_mat_M)