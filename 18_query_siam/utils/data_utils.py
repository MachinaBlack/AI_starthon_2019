import re
import html
import string
import numpy as np
from konlpy.tag import Mecab


PAD = "[PAD]"
PAD_IND = 0

mecab = Mecab()

def get_all_chars():
    koreans = [chr(i) for i in range(44032, 55204)] # 가-힣
    korean_chars = [chr(i) for i in range(ord("ㄱ"), ord("ㅣ") + 1)] # ㄱ-ㅎ, ㅏ-ㅣ
    alphabets = list(string.ascii_letters)
    digits = list(string.digits)
    return [PAD, " "] + koreans + alphabets


# build char vocabulary
vocabs = get_all_chars()
ind2vocab = {ind: char for ind, char in enumerate(vocabs)}
vocab2ind = {char: ind for ind, char in enumerate(vocabs)}

_vocabs = "[^" + "".join(vocabs[1:]) + "]"


def prepro_text(text):
    text = html.unescape(text)
    # text = re.sub(_vocabs, " ", text)
    text = text.lower()
    text = " ".join(mecab.morphs(text))

    return re.sub("\s+", " ", text).strip()



def text2ind(text, max_len, raw_text=False):
    if raw_text:
        text = prepro_text(text)
    return np.asarray(list(map(lambda char: vocab2ind[char], text))[:max_len] + \
                      [vocab2ind[PAD] for _ in range(max((max_len - len(text)), 0))])

def text2ind_sp(text, max_len, sp, raw_text=False):
    if raw_text:
        text = prepro_text(text)
    ids = sp.encode_as_ids(text)
    ids = ids[:max_len-2]
    ids = [1] + ids + [2]
    return np.asarray(ids[:max_len] + [2 for _ in range(max((max_len - len(ids)), 0))])

def ind2text(inds):
    return "".join(map(lambda ind: ind2vocab[ind] if ind >= 0 else "", inds))
