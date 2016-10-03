#coding:utf-8
from c_xml_parse import Xml_Parse_base
from xml.dom import minidom
import re
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
    # to gain the 'Other' relation mention from ACE dataset
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


class ACE_parse(Xml_Parse_base):
    # Extract and save relation mation(relation_type = "other") from ACE data
    def __init__(self, save_path, path_dic, language):
        self.path_dic = path_dic
        self.save_path = save_path
        self.language = language

    def __extract_other(self, file_name):
        # parse xml from ACE data and trun the raw data into ACE_neg form
        # this is for relation type "other"
        doc = minidom.parse(file_name)
        root = doc.documentElement
        R_element = ACE_neg()

        R_element.file_name = self.get_attrvalue(root, 'URI')
        relation_nodes = self.get_xmlnode(root, 'relation')
        entity_nodes = self.get_xmlnode(root, 'entity')

        for node in entity_nodes:
            entity_mention = self.get_xmlnode(node, 'entity_mention')
            temp_pos = []
            for mention_node in entity_mention:
                entity_extent = self.get_xmlnode(mention_node, 'charseq')
                pos = [0, 0]
                pos[0] = int(self.get_attrvalue(entity_extent[0], 'START'))
                pos[1] = int(self.get_attrvalue(entity_extent[0], 'END'))
                temp_pos.append(pos)
            R_element.entity_pos.append(temp_pos)

        for node in relation_nodes:
            mention_nodes = self.get_xmlnode(node, 'relation_mention')

            mention_pos = []
            arg1_pos = []
            arg2_pos = []

            for mention_node in mention_nodes:
                # gain the attribute info of mention
                mention_extent = self.get_xmlnode(mention_node, 'charseq')
                pos2 = [0, 0]
                pos2[0] = int(self.get_attrvalue(mention_extent[0], 'START'))
                pos2[1] = int(self.get_attrvalue(mention_extent[0], 'END'))
                mention_pos.append(pos2)

                argument_nodes = self.get_xmlnode(mention_node, 'relation_mention_argument')
                for argument_node in argument_nodes:
                    argument_extent = self.get_xmlnode(argument_node, 'charseq')
                    pos3 = [0, 0]
                    pos3[0] = int(self.get_attrvalue(argument_extent[0], 'START'))
                    pos3[1] = int(self.get_attrvalue(argument_extent[0], 'END'))
                    if self.get_attrvalue(argument_node, 'ROLE') == 'Arg-1':
                        arg1_pos.append(pos3)
                    elif self.get_attrvalue(argument_node, 'ROLE') == 'Arg-2':
                        arg2_pos.append(pos3)

            R_element.arg2_pos.append(arg2_pos)
            R_element.arg1_pos.append(arg1_pos)
            R_element.mention_pos.append(mention_pos)
        return R_element

    def get_other(self):
        # read the file from directory and process data, at last , save the data as our form
        # this is for relation type "other"
        f_out = open(self.save_path, 'w')
        f_names = os.listdir(self.path_dic)
        for name in f_names:
            if "apf.xml" not in name or 'score' in name:
                continue
            # extract mention and turn the mention into form we required
            R = self.__extract_other(self.path_dic + "/" + name)
            R.get_raw_txt(self.path_dic + "/")
            R.fenjv()
            other = R.get_other_type()
            # different language has different save form
            if self.language == "ch":
                for i in range(0, len(other)):
                    other[i] = other[i].replace('\t', ' ')
                    f_out.write(other[i].replace('\n', ' ') + '\tother\tother\n')
            else:
                for i in range(0, len(other)):
                    other[i] = other[i].replace('\t', ' ')
                    f_out.write(other[i].replace('\n', ' ') + '\n')
        f_out.close()

    def __extract_info(self, file_name):
        # parse xml from ACE data and turn the raw data into ACE_neg form
        # this is for annotated relation type
        R = []
        doc = minidom.parse(file_name)
        root = doc.documentElement

        relation_nodes = self.get_xmlnode(root, 'relation')
        for node in relation_nodes:
            relation_type = self.get_attrvalue(node, 'TYPE')
            relation_sub_type = self.get_attrvalue(node, 'SUBTYPE')
            mention_nodes = self.get_xmlnode(node, 'relation_mention')
            for mention_node in mention_nodes:
                R_element = ACE_info()
                R_element.type = relation_type
                R_element.sub_type = relation_sub_type

                # gain the attribute info of mention
                mention_extent = self.get_xmlnode(mention_node, 'charseq')
                R_element.mention_pos[0] = int(self.get_attrvalue(mention_extent[0], 'START'))
                R_element.mention_pos[1] = int(self.get_attrvalue(mention_extent[0], 'END'))
                R_element.mention = self.get_nodevalue(mention_extent[0])

                argument_nodes = self.get_xmlnode(mention_node, 'relation_mention_argument')
                for argument_node in argument_nodes:
                    if self.get_attrvalue(argument_node, 'ROLE') == 'Arg-1':
                        # gain the attribute info of arg1
                        R_element.arg1_pos[0] = int(self.get_attrvalue(self.get_xmlnode(argument_node, 'charseq')[0], 'START'))
                        R_element.arg1_pos[1] = int(self.get_attrvalue(self.get_xmlnode(argument_node, 'charseq')[0], 'END'))
                        #relation_arg1 = get_nodevalue(get_xmlnode(argument_node, 'charseq')[0]).encode('utf-8', 'ignore')
                        #print(R_element.arg1_pos[0],R_element.arg1_pos[1])
                    elif self.get_attrvalue(argument_node, 'ROLE') == 'Arg-2':
                        # gain the attribute info of age2
                        R_element.arg2_pos[0] = int(self.get_attrvalue(self.get_xmlnode(argument_node, 'charseq')[0], 'START'))
                        R_element.arg2_pos[1] = int(self.get_attrvalue(self.get_xmlnode(argument_node, 'charseq')[0], 'END'))
                        #relation_arg2 = get_nodevalue(get_xmlnode(argument_node, 'charseq')[0]).encode('utf-8', 'ignore')
                        #print(R_element.arg2_pos[0],R_element.arg2_pos[1])
                R_element.combine()
                R.append(R_element)
        return R

    def get_info(self):
        # read the file from directory and process data, at last , save the data as our form
        # this is for annotated relation type
        f_out = open(self.save_path, 'w')
        f_names = os.listdir(self.path_dic)
        for name in f_names:
            if "apf.xml" not in name or 'score' in name:
                continue
            R = self.__extract_info(self.path_dic +"/" + name)
            for i in range(0, len(R)):
                R[i].out_form = R[i].out_form.replace('\n', ' ')
                f_out.write(R[i].out_form.encode('utf-8'))
                f_out.write('\n')
        f_out.close()