import re
import string
from multiprocessing import Pool, cpu_count

import nltk
import numpy as np
import pandas as pd
import torch
from gensim.models import KeyedVectors
from torch.autograd import Variable

from net.Net import Net
from preprocessing import pre_process

fast_text = KeyedVectors.load("data/wiki.ar.genism")
max_sentence_length = 25


def extract_vectors(tweet):
    words = [re.sub(r'(.)\1+', r'\1', word) for word in nltk.word_tokenize(tweet) if
             "." not in word and word not in string.punctuation and "،" not in word]
    # words = [word if word in self.fast_text.wv else "unk" for word in words]
    words = [word for word in words if word in fast_text.wv]
    temp = torch.from_numpy(fast_text[words])
    l = len(words)
    if l > max_sentence_length:
        return temp[:max_sentence_length, :]
    if l == max_sentence_length:
        return temp
    padding_size = max_sentence_length - l
    t = torch.zeros(padding_size, 300)
    return torch.cat((temp, t))


def to_module(state_dict):
    new_state_dict = dict()
    for key in state_dict.keys():
        new_name = key[key.index(".") + 1:]
        new_state_dict[new_name] = state_dict[key]
    return new_state_dict


state = torch.load("net/checkpoint.pth.tar")
temp = state["state_dict"]
state_dict = to_module(temp)
net = Net([3, 4, 5]).cuda()
net.load_state_dict(state_dict)
net = net.eval()
tweets = pd.read_csv("data/Driving_Data_Cleaned_with_hashtag.txt", sep=r"\s?\|\|\s?", skip_blank_lines=True, engine='python',
                     encoding="utf-8")
# tweets = tweets.iloc[:500, :]
tweets = pre_process(tweets)
X, y = tweets["tweet"].astype(np.str), tweets["sentiment"].astype(np.int)
labels = torch.from_numpy(y.values).cuda()
with Pool(cpu_count()) as p:
    tweets = p.map(extract_vectors, X.values)
temp = torch.zeros(25, 300)
for i in range(len(tweets)):
    temp = torch.cat((temp, tweets[i]))
tweets = temp[26:, :]
tweets = tweets.unsqueeze(0)
tweets = Variable(tweets.cuda())
y = net(tweets)
_, predicted = torch.max(y, 1)
acc = (predicted.data == labels).sum() / labels.shape[0] * 100
print("acc = {}".format(acc))
