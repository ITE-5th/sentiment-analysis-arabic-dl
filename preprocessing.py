import re

import numpy as np
from pyarabic.araby import strip_tashkeel


def pre_process(data):
    data = data.dropna(how="any")
    data.loc[:, "sentiment"] = data.loc[:, "sentiment"].apply(lambda x: int(x.lower().strip() == "yes")).astype(np.int)
    data = pre_process_tweet(data)
    data = data.drop_duplicates(subset="tweet")
    return data


def pre_process_tweet(data):
    data.loc[:, "tweet"] = data.loc[:, "tweet"].apply(lambda x: x.strip())
    data.loc[:, "tweet"] = data.loc[:, "tweet"].apply(lambda x: strip_tashkeel(x))
    data.loc[:, "tweet"] = data.loc[:, "tweet"].apply(lambda x: re.sub("[أإآ]", "ا", x))
    data.loc[:, "tweet"] = data.loc[:, "tweet"].apply(lambda x: re.sub("#", "", x))
    data.loc[:, "tweet"] = data.loc[:, "tweet"].apply(lambda x: re.sub("_", " ", x))
    data.loc[:, "tweet"] = data.loc[:, "tweet"].apply(
        lambda x: re.sub("([1-9] |1[0-9]| 2[0-9]|3[0-1])(.|-)([1-9] |1[0-2])(.|-|)(20|14)[0-9][0-9]", " ", x))
    # TODO: are we sure that we should delete all the tweets with any english letter?
    # data = data.loc[~data.loc[:, "tweet"].str.contains("[a-zA-Z]"), :]
    data = data.loc[~data.loc[:, "tweet"].str.contains("http"), :]
    return data
