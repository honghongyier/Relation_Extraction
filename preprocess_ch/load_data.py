#coding:utf-8
import re
from pyltp import Segmentor
from preprocess_ch.data2vector import load_word_embedding
import csv


def seg_initialize(model_path, lexicon_path):
    print "load segment data..."
    segmentor = Segmentor()
    segmentor.load_with_lexicon(model_path, lexicon_path)
    return segmentor


def load_seg_data(data_path, vocab, seg_model, filename, doc_len=50):
    print "load chinese data..."
    file_raw = open(data_path)
    file_out = file(filename, "wb")
    csv_writer = csv.writer(file_out)
    relation = dict()
    triple = ["s", "r", "o"]
    data_len = 0
    datas = list()
    for line in file_raw:
        data_len += 1
        line = line.replace("\n", "")
        line = line.split("\t")
        line[0] = line[0].replace(" ", "")
        e1 = re.findall('<e1>(.*?)</e1>', line[0], re.S)
        e2 = re.findall('<e2>(.*?)</e2>', line[0], re.S)
        if len(e1) == 0:
            triple[0] = "bad"
            continue
        if len(e2) == 0:
            triple[2] = "bad"
            continue
        triple[0] = e1[0]
        triple[1] = line[1].replace("cr", "")
        if not relation.has_key(triple[1]):
            relation[triple[1]] = len(relation)
        triple[2] = e2[0]
        part1 = re.findall('^(.*?)<', line[0], re.S)
        part2 = re.findall('/e\d*>(.*?)<e\d*', line[0], re.S)
        part3 = re.findall('</e\d*>.*?<.*?><.*?>(.*?)$', line[0], re.S)
        total = ""
        if len(part1) > 0:
            seg_1 = seg_model.segment(part1[0])
            total += "\t".join(seg_1)
        if len(part2) > 0:
            seg_2 = seg_model.segment(part2[0])
            total += "\t"+triple[0]+"\t"+"\t".join(seg_2)+"\t"+triple[2]
        if len(part3) > 0:
            seg_3 = seg_model.segment(part3[0])
            total += "\t"+"\t".join(seg_3)
        total = total.split("\t")
        triple[0] = triple[0].split("\t")
        triple[2] = triple[2].split("\t")
        # get the PF value
        PF = GetPF(total, triple)
        # total to id, format: word_id, PF_1_id, PF_2_id, relation_id
        '''
        for i in range(len(total)):
            if vocab.has_key(total[i]):
                total[i] = [vocab[total[i]], PF['1'][i], PF['2'][i]]
            else:
                total[i] = [len(vocab), PF['1'][i], PF['2'][i]]
        '''
        # len(vocab) = 'unk', len(vocab)+1 = zero
        # print PF, total
        record = list()
        if len(total) > doc_len:
            start = (len(total)-doc_len)/2
            for i in range(start, start+doc_len):
                PF['1'][i] = int(PF['1'][i]) + doc_len - 1
                if PF['1'][i] < 0:
                    PF['1'][i] = 0
                PF['2'][i] = int(PF['2'][i]) + doc_len - 1
                if PF['2'][i] < 0:
                    PF['2'][i] = 0
                if vocab.has_key(total[i]):
                    record.append(vocab[total[i]])
                else:
                    record.append(len(vocab))   # 等价于“UNK”
            record.extend(PF['1'][start:start+doc_len])
            record.extend(PF['2'][start:start+doc_len])
            record.append(relation[triple[1]])
            if len(record) != doc_len*3+1:
                print len(record)
                print "error in", line
            csv_writer.writerow(tuple(record))
        else:
            for i in range(len(total)):
                PF['1'][i] = int(PF['1'][i]) + doc_len - 1
                PF['2'][i] = int(PF['2'][i]) + doc_len - 1
                if vocab.has_key(total[i]):
                    record.append(vocab[total[i]])
                else:
                    record.append(len(vocab))   # 等价于"UNK"
            for i in range(doc_len-len(total)):
                record.append(len(vocab)+1) # 补零
                PF['1'].append(0)
                PF['2'].append(0)
            record.extend(PF['1'])
            record.extend(PF['2'])
            record.append(relation[triple[1]])
            if len(record) != doc_len*3+1:
                print "error in", line
            csv_writer.writerow(tuple(record))
        #datas.append(total)
    file_raw.close()
    file_out.close()
    return data_len, relation


def GetPF(word, triple):
    PF = {'1': [], '2': []}  # '1'是距离triple[0]的距离，'1'是距离triple[2]的距离
    # 获取index_1的值
    if len(triple[0]) == 1:
        try:
            index_1 = word.index(triple[0][0])
            for i in range(0, len(word)):
                PF['1'].append(i - index_1)
        except ValueError:
            index_1 = -1
            for i in range(0, len(word)):
                if triple[0][0] in word[i]:
                    index_1 = i
                    break
            if index_1 != -1:
                for i in range(0, len(word)):
                    PF['1'].append(i - index_1)
            else:
                for i in range(0, len(word)):
                    PF['1'].append("-1")
    else:
        index_1 = word.index(triple[0][0])
        for i in range(0, len(word)):
            if (i > index_1 and i < (index_1 + len(triple[0]) + 1)):
                PF['1'].append(0)
            elif (i > (index_1 + len(triple[0]))):
                PF['1'].append(i - index_1 - len(triple[0]))
            else:
                PF['1'].append(i - index_1)

    # 获取index_2的值
    if len(triple[2]) == 1:
        try:
            index_2 = word.index(triple[2][0])
            for i in range(0, len(word)):
                PF['2'].append(i - index_2)
        except ValueError:
            index_2 = -1
            for i in range(0, len(word)):
                if triple[2][0] in word[i]:
                    index_2 = i
                    break
            if index_2 != -1:
                for i in range(0, len(word)):
                    PF['2'].append(i - index_2)
            else:
                for i in range(0, len(word)):
                    PF['2'].append("-1")
    else:
        index_2 = word.index(triple[2][0])
        for i in range(0, len(word)):
            if i > index_2 and i < (index_2 + len(triple[2]) + 1):
                PF['2'].append(0)
            elif i > (index_2 + len(triple[2])):
                PF['2'].append(i - index_2 - len(triple[2]))
            else:
                PF['2'].append(i - index_2)

    return PF



if __name__ == "__main__":
    raw_data_path = "../Raw_Data/ten_relations"
    seg_model_path = "../data_tool/seg_tool/cws.model"
    dic_model_path = "../data_tool/seg_tool/vocab.txt"
    vocab_path = "../data_tool/word2vec/vectors.txt"
    seg_model = seg_initialize(seg_model_path, dic_model_path)
    vocab, _, _ = load_word_embedding(vocab_path)
    data_len = load_seg_data(raw_data_path, vocab, seg_model, "../out/data_ch.csv")
    # write to csv
    # print datas[0][0]
