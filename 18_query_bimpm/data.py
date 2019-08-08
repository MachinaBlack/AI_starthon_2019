import html
import math
import os
import pickle
import re
from shutil import copyfile
from itertools import combinations

import sentencepiece as sp
import torch
from konlpy.tag import Mecab

mecab = Mecab()


def read_questions(file, label_file=None):
    qs = []
    with open(file) as f:
        questions = [line.split('\t') for line in f.read().splitlines()]
    
    if label_file:
        with open(label_file) as f:
            labels = [int(label) for label in f.read().splitlines()]

    if label_file:
        for (_, question1, question2), label in zip(questions, labels):
            qs.append((question1, question2, label))
    else:
        for (_, question1, question2) in questions:
            qs.append(question1)
            qs.append(question2)

    return qs


def normalize(text):
    text = html.unescape(text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()  # strips consecutive spaces
    return text


def morph(text):
    return ' '.join(mecab.morphs(text))


class Dataset:
    def __init__(self, question_file_path, label_file_path, vocab, feature, mode='train'):
        self.question_file_path = question_file_path
        self.label_file_path = label_file_path
        self.vocab = vocab
        self.feature = feature
        self.mode = mode

        # questions in raw text
        self.questions1 = []
        self.questions2 = []
        self.labels = None
        # sentence embeddings
        self.embeddings1 = None
        self.embeddings2 = None
        # feature vectors
        self.features = None

        self.load_data(question_file_path, label_file_path)
        self.get_sentence_embeddings()
        self.extract_features()

    def __len__(self):
        assert len(self.questions1) == len(self.questions2)
        return len(self.questions1)

    def __getitem__(self, idx):
        items = torch.tensor(self.embeddings1[idx]), torch.tensor(self.embeddings2[idx]), \
                torch.tensor(self.features[idx]).float()
        if self.labels is not None:
            items += (torch.tensor(self.labels[idx]), )

        return items

    def load_data(self, question_file, label_file):
        with open(question_file) as f:
            questions = [line.split('\t') for line in f.read().splitlines()]

            for (_, question1, question2) in questions:
                self.questions1.append(question1)
                self.questions2.append(question2)

        if label_file:
            with open(label_file) as f:
                labels = [int(label) for label in f.read().splitlines()]
            self.labels = labels

    def get_sentence_embeddings(self):
        self.embeddings1 = [self.vocab.embed(morph(normalize(q))) for q in self.questions1]
        self.embeddings2 = [self.vocab.embed(morph(normalize(q))) for q in self.questions2]

    def extract_features(self):
        self.features = [self.feature.extract_feature(q1, q2) for q1, q2 in zip(self.questions1, self.questions2)]


class Vocabulary:
    OOV_IND = 0
    BOS_IND = 1
    EOS_IND = 2
    PAD_IND = 3

    def __init__(self, config):
        self.vocabs = []
        self.vocab2idx = None
        self.vocab_file_prefix = config.vocab_file_prefix
        self.vocab_size = config.vocab_size
        self.max_sequence_len = config.max_sequence_len
        self.sp_processor = None

        if config.mode == 'train':
            # in the train mode, we create a vocab file from the data
            # in test mode, we will load the object from the saved file thus we don't have to create one

            self.train_vocab(config)
            self.load_vocab('./')

    def train_vocab(self, config):
        qs = []
        train_question_file_path = os.path.join(config.data_dir, config.train_file_name)
        qs += read_questions(train_question_file_path)
#        validation_question_file_path = os.path.join(config.data_dir, config.validation_file_name)
#        qs += read_questions(validation_question_file_path)
        qs = map(lambda q: morph(normalize(q)), qs)

        questions_file_name = 'questions.txt'
        with open(questions_file_name, 'w') as f:
            for question in qs:
                f.write("%s\n" % question)

        cmd = '--input={} --model_prefix={} --vocab_size={} --pad_id={}'.format(
            questions_file_name, self.vocab_file_prefix, self.vocab_size, self.PAD_IND)
        sp.SentencePieceTrainer.Train(cmd)

    def save_vocab(self, path):
        copyfile(self.vocab_file_prefix + '.model', os.path.join(path, self.vocab_file_prefix + '.model'))
        copyfile(self.vocab_file_prefix + '.vocab', os.path.join(path, self.vocab_file_prefix + '.vocab'))

    def load_vocab(self, path):
        self.sp_processor = sp.SentencePieceProcessor()
        self.sp_processor.Load(os.path.join(path, self.vocab_file_prefix + '.model'))

        with open(os.path.join(path, self.vocab_file_prefix + '.vocab'), encoding='utf-8') as f:
            for line in f:
                tab_idx = line.find('\t')
                vocab = line[:tab_idx]
                self.vocabs.append(vocab)
        self.vocab2idx = {subword: idx for idx, subword in enumerate(self.vocabs)}

        print(f'vocab loaded: size {len(self.vocabs)}')

    def vocab_size(self):
        return len(self.vocabs)

    def tokenize(self, text):
        return self.sp_processor.EncodeAsPieces(text)

    def indexOf(self, vocab):
        return self.vocab2idx.get(vocab, Vocabulary.OOV_IND)

    def embed(self, text):
        tokenized = self.tokenize(text)
        embedding = ([Vocabulary.BOS_IND] + [self.indexOf(token) for token in tokenized] +
                     [Vocabulary.EOS_IND])[:self.max_sequence_len]
        padding = [self.PAD_IND for _ in range(max((self.max_sequence_len - len(embedding)), 0))]
        return embedding + padding


class Feature:
    def __init__(self, config):
        self.feature_data_file_prefix = config.feature_data_file_prefix
        self.ngram_max = config.ngram_max
        self.nterm_max = config.nterm_max

        self.idf = {}
        self.pw_one_side = {}
        self.pw_both_side = {}

        if config.mode == 'train':
            # in the train mode, we create a feature data file from the data
            # in test mode, we will load the object from the saved file thus we don't have to create one
            self.calculate_idf(config)
            self.calculate_power_word(config, 50)

        # get features size from dummy texts
        self.size = len(self.extract_feature('', ''))

    def calculate_idf(self, config):
        qs = []
        train_question_file_path = os.path.join(config.data_dir, config.train_file_name)
        qs += read_questions(train_question_file_path)
        validation_question_file_path = os.path.join(config.data_dir, config.validation_file_name)
        qs += read_questions(validation_question_file_path)
        qs = map(lambda q: morph(normalize(q)), qs)

        count = {}
        for question in qs:
            for word in question.split('\s'):
                count[word] = count.get(word, 0) + 1
        num_words = len(count)

        for word in count:
            self.idf[word] = math.log(num_words / (count[word] + 1.)) / math.log(2.)

        print(f'idf calculated: size {len(self.idf)}')

    def calculate_power_word(self, config, num_least=50):
        qs = []
        train_question_file_path = os.path.join(config.data_dir, config.train_file_name)
        train_label_file_path = os.path.join(config.data_dir, config.train_label_file_name)
        qs += read_questions(train_question_file_path, train_label_file_path)
        validation_question_file_path = os.path.join(config.data_dir, config.validation_file_name)
        validation_label_file_path = os.path.join(config.data_dir, config.validation_label_file_name)
        qs += read_questions(validation_question_file_path, validation_label_file_path)
        qs = map(lambda t: (morph(normalize(t[0])), morph(normalize(t[1])), t[2]), qs)

        count = {}
        for question1, question2, label in qs:
            q1_set = set(mecab.morphs(question1))
            q2_set = set(mecab.morphs(question2))

            # count the total occurence
            for word in q1_set | q2_set:
                if word not in count:
                    count[word] = {'total': 0, 'one_side': 0, 'one_side_nodup': 0,
                            'both_side': 0, 'both_side_dup': 0}
                count[word]['total'] += 1

            # only on one side
            for word in q1_set ^ q2_set:
                count[word]['one_side'] += 1
                if label == 0:
                    count[word]['one_side_nodup'] += 1

            # both side
            for word in q1_set & q2_set:
                count[word]['both_side'] += 1
                if label == 0:
                    count[word]['both_side_dup'] += 1

        self.pw_one_side = {}
        self.pw_both_side = {}
        for word in count:
            # minimum total count
            if count[word]['one_side'] > num_least:
                self.pw_one_side[word] = count[word]['one_side_nodup'] / count[word]['one_side']
            if count[word]['both_side'] > num_least:
                self.pw_both_side[word] = count[word]['both_side_dup'] / count[word]['both_side']
                
        print(f'power word calculated: size {len(self.pw_one_side)}, {len(self.pw_both_side)}')

    def save_data(self, path):
        with open(os.path.join(path, self.feature_data_file_prefix + '.pickle'), 'wb') as f:
            dict = {
                'idf': self.idf,
                'pw_one_side': self.pw_one_side,
                'pw_both_side': self.pw_both_side
            }
            pickle.dump(dict, f)

    def load_data(self, path):
        with open(os.path.join(path, self.feature_data_file_prefix + '.pickle'), 'rb') as f:
            dic = pickle.load(f)
            self.idf = dic['idf']
            self.pw_one_side = dic['pw_one_side']
            self.pw_both_side = dic['pw_both_side']

        print(f'idf loaded: size {len(self.idf)}')
        print(f'power one side loaded: size {len(self.pw_one_side)}')
        print(f'power both side loaded: size {len(self.pw_both_side)}')

    def extract_feature(self, text1, text2):
        feature = []
        # morphed and normalized tokens
        mnt1, mnt2 = morph(normalize(text1)), morph(normalize(text2))

        # various lengths
        def lengths(t1, t2):
            l1, l2 = len(t1), len(t2)
            return [l1, l2, abs(l1 - l2), min(l1, l2), max(l1, l2)]
        feature += lengths(text1, text2)
        feature += lengths(normalize(text1), normalize(text2))
        feature += lengths(mnt1, mnt2)

        # jaccard similarity of unique sets
        m1, m2 = mnt1.split(' '), mnt2.split(' ')
        s1, s2 = set(m1), set(m2)
        feature.append(len(s1 & s2) / len (s1 | s2))

        # idf weighted
        common_words_in_m1 = sum([self.idf.get(w, 0) for w in m1 if w in m2])
        common_words_in_m2 = sum([self.idf.get(w, 0) for w in m2 if w in m1])
        total = sum(self.idf.get(w, 0) for w in m1) + sum(self.idf.get(w, 0) for w in m2)
        if total == 0:
            # to avoid dividing by zero
            feature.append(0)
        else:
            feature.append((common_words_in_m1 + common_words_in_m2) / total)

        def dice_distance(a, b):
            a = set(a)
            b = set(b)

            if len(a) == 0 and len(b) == 0:
                return -1
            else:
                return (2. * float(len(a & b))) / (len(a) + len(b))

        # n-grams
        def ngram(words, n):
            return list(zip(*[words[i:] for i in range(n)]))
        for n in range(1, self.ngram_max + 1):
            ng1 = ngram(m1, n)
            ng2 = ngram(m2, n)
            feature.append(dice_distance(ng1, ng2))

        # n-terms
        for n in range(1, self.nterm_max + 1):
            nt1 = list(combinations(m1, n))
            nt2 = list(combinations(m2, n))
            feature.append(dice_distance(nt1, nt2))

        def power_rate(m1, m2):
            s1 = set(m1)
            s2 = set(m2)

            dup_rate = 1
            share_bw = [w for w in (s1 & s2) if (w in self.pw_both_side)]
            for w in share_bw:
                dup_rate *= 1 - self.pw_both_side[w]
            dup_rate = 1 - dup_rate
 
            nodup_rate = 1
            diff_ow = [w for w in (s1 ^ s2) if (w in self.pw_one_side)]
            for w in diff_ow:
                nodup_rate *= 1 - self.pw_one_side[w]
            nodup_rate = 1 - nodup_rate

            return [dup_rate, nodup_rate]
        feature += power_rate(mnt1, mnt2)

        return feature
