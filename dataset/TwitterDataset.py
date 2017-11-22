import re
import string
from multiprocessing import Pool, cpu_count
import nltk
import numpy as np
import pandas as pd
import torch
from gensim.models import KeyedVectors
from torch.utils.data.dataset import Dataset
from preprocessing import pre_process


class TwitterDataset(Dataset):
    def __init__(self, root_path: str):
        self.fast_text = KeyedVectors.load(root_path + "/wiki.ar.genism")
        # self.fast_text.wv["unk"] = np.random.normal(size=300)
        tweets = pd.read_csv(root_path + "/Driving_Data_Cleaned_with_hashtag.txt", sep=r"\s?\|\|\s?",
                             skip_blank_lines=True, engine='python', encoding="utf-8")
        tweets = pre_process(tweets)
        X, y = tweets["tweet"].astype(np.str), tweets["sentiment"].astype(np.int)
        self.labels = torch.from_numpy(y.values)
        self.max_sentence_length = 35
        self.unk = np.random.normal(size=300).astype(np.float)
        print("now extracting")
        with Pool(cpu_count()) as p:
            self.tweets = p.map(self.extract_vectors, X.values)
        # self.tweets = list(map(self.extract_vectors, X.values))

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        return self.tweets[index], self.labels[index]

    def extract_vectors(self, tweet):
        words = [re.sub(r'(.)\1+', r'\1', word) for word in nltk.word_tokenize(tweet) if
                 "." not in word and word not in string.punctuation and "،" not in word]
        words = [self.fast_text[word].astype(np.float) if word in self.fast_text.wv else self.unk for word in words]
        words = np.array(words).astype(np.float)
        # words = [word for word in words if word in self.fast_text.wv]
        # temp = torch.from_numpy(self.fast_text[words])
        temp = torch.from_numpy(words).float()
        l = len(words)
        if l > self.max_sentence_length:
            return temp[:self.max_sentence_length, :]
        if l == self.max_sentence_length:
            return temp
        padding_size = self.max_sentence_length - l
        t = torch.zeros(padding_size, 300).float()
        return torch.cat((temp, t))
