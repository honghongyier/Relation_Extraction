#coding:utf-8
import numpy as np

class Relation:
    # Definition a class to turn relation mention to embedding
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