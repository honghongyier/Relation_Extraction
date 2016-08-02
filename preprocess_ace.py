#coding:utf-8
'''
This is a preprocess python file for ACE2005 dataset and SemEval dataset
You can download this dataset from LDC


As for the word2vec embedding, we use pre-trained word embedding
Which could download here: https://code.google.com/p/word2vec/

'''
import numpy as np
from ACE_Process import *
from Open_Tool import Open_tool


if __name__ == "__main__":
    language = "en"  # you can change en to "ch" if the input in a chinese data set

    # First-Part: extraction ACE mention from raw dataset
    # to gain all the mention from files , you should change the file name by yourself
    other_save_path = "out/ace2005/other_en_wl.txt"
    mention_save_path = "out/ace2005/en_wl.txt"
    directory = "Raw_Data/ace_2005_td_v7/data/English/wl/adj"

    ace_parse = ACE_parse(other_save_path, directory, language)
    ace_parse.get_other()
    ace_parse.get_info()


    # Second-Part: turn the relation mention into wordembedding, the input support mention produced from first

    # change the mention into embedding
    open_tool = Open_tool()

    # load word2vec
    vector_path = "out/GoogleNews-vectors-negative300.bin"
    vector_dim = 300
    vector = open_tool.load_word2vec(vector_path, True)

    # load your position embedding matrix or generate a new one
    PF_dim = 50
    PF_size = 100
    PF_R1 = np.load("out/PF_R1.npy")
    PF_R2 = np.load("out/PF_R2.npy")

    '''
    #PF_R1, PF_R2 = open_tool.get_pfmartrix(PF_dim, PF_size, "out/")
    '''


    # preprocessing start
    # code below also work on SEM dataset
    f_path = "out/ace2005"  # mention path
    out_dic = "out/ace2005/Data300_PF50/"
    R, type = open_tool.sentence2vec(vector, vector_dim, f_path, out_dic, PF_R1, PF_R2, PF_dim, PF_size, language)

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
