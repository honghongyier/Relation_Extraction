#coding:utf-8
from numpy import random
import numpy as np

def load_word_embedding(embedding_path):
    embed = list()
    vocab = dict()
    embed_file = open(embedding_path)
    for line in embed_file:
        line = line.replace("\n", "")
        line = line.split(" ")
        if len(line) < 5:
            continue
        #embed[line[0]] = []
        temp = []
        vocab[line[0]] = len(vocab)
        for i in range(1, len(line)-1):
            temp.append(float(line[i]))
            #embed[line[0]].append(float(line[i]))
        embed.append(temp)
    print len(embed), len(embed[0])
    dim = len(line)-2
    embed.append(list(random.random(dim)))  # "UNK"
    embed.append([0.0]*dim)  # zero embedding
    return vocab, np.array(embed), dim

def generate_position_embedding(scale=50, dim=5):
    PF_R1 = random.random(size=(scale*2, dim))  # position scale -(scale-1) ~ scale
    PF_R2 = random.random(size=(scale*2, dim))
    np.save("out/PF_R1.npy", PF_R1)
    np.save("out/PF_R2.npy", PF_R2)
    return PF_R1, PF_R2




