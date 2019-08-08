from konlpy.tag import Mecab
from utils.data_utils import prepro_text
import math
import pickle

mecab = Mecab()


class Feature:
    def __init__(self):
        self.idf = {}
        feats = self.feat_test()
        self.size = len(feats)

    def extract_feature(self, q1, q2):
        # q1, q2: raw questions
        features = []
        raw_q1 = q1
        raw_q2 = q2
        q1 = prepro_text(q1)
        q2 = prepro_text(q2)
        q1_morphs = mecab.morphs(q1)
        q2_morphs = mecab.morphs(q2)

        def length(q1, q2):
            features = []
            features.append(len(q1))
            features.append(len(q2))
            features.append(abs(len(q1) - len(q2)))
            features.append(min(len(q1), len(q2)) / max(len(q1), len(q2)))
            return features
        features += length(raw_q1, raw_q2)
        features += length(q1, q2)
        features += length(q1_morphs, q2_morphs)

        def word_match(q1, q2):
            features = []

            # unique word list
            s1 = set(q1)
            s2 = set(q2)
            features.append(len(s1 & s2) / len(s1 | s2))

            # multiple occurence
            cnt = 0
            for w in q1:
                if w in s2:
                    cnt += 1
            for w in q2:
                if w in s1:
                    cnt += 1
            features.append(cnt / (len(q1) + len(q2)))

            # idf weight
            sum_shared_word_in_q1 = sum([self.idf.get(w, 0) for w in q1 if w in s2])
            sum_shared_word_in_q2 = sum([self.idf.get(w, 0) for w in q2 if w in s1])
            try:
                sum_tol = sum(self.idf.get(w, 0) for w in q1) + sum(self.idf.get(w, 0) for w in q2)
                features.append((sum_shared_word_in_q1 + sum_shared_word_in_q2) / sum_tol)
            except ZeroDivisionError:
                features.append(0)

            return features
        features += word_match(q1_morphs, q2_morphs)

        def set_dist(q1, q2):
            features = []
            for n in range(1, 4):
                q1_ngrams = NgramUtil.ngrams(q1, n)
                q2_ngrams = NgramUtil.ngrams(q2, n)
                features.append(DistanceUtil.dice_dist(q1_ngrams, q2_ngrams))
            return features
        features += set_dist(q1_morphs, q2_morphs)

        return features

    def load_idf(self, path):
        with open(path, 'rb') as fp:
            feature = pickle.load(fp)
            self.idf = feature.idf

    def init_idf(self, path):
        idf = {}
        q_set = set()
        with open(path) as f:
            # data 읽어오는 부분
            data = f.read().splitlines()
            data = [line.split("\t") for line in data]

            _, a_seqs, b_seqs = list(zip(*data))

            for a_seq, b_seq in zip(a_seqs, b_seqs):
                q1 = a_seq
                q1_morphs = mecab.morphs(q1)
                q2 = b_seq
                q2_morphs = mecab.morphs(q2)
                if q1 not in q_set:
                    q_set.add(q1)
                    for word in q1_morphs:
                        idf[word] = idf.get(word, 0) + 1
                if q2 not in q_set:
                    q_set.add(q2)
                    for word in q2_morphs:
                        idf[word] = idf.get(word, 0) + 1
            num_docs = len(q_set)
            for word in idf:
                idf[word] = math.log(num_docs / (idf[word] + 1.)) / math.log(2.)
            self.idf = idf

    def feat_test(self):
        q1 = '오늘 강남에서 맛있는 스파게티를 먹었다.'
        q2 = '강남에서 먹었던 오늘의 스파게티는 맛있었다.'

        print("mecab example")
        print(mecab.morphs(q1))
        print(mecab.nouns(q1))
        print(mecab.pos(q1))

        feats = self.extract_feature(q1, q2)
        print(feats)
        return feats

class NgramUtil(object):

    def __init__(self):
        pass

    @staticmethod
    def unigrams(words):
        """
            Input: a list of words, e.g., ["I", "am", "Denny"]
            Output: a list of unigram
        """
        assert type(words) == list
        return words

    @staticmethod
    def bigrams(words, join_string, skip=0):
        """
           Input: a list of words, e.g., ["I", "am", "Denny"]
           Output: a list of bigram, e.g., ["I_am", "am_Denny"]
        """
        assert type(words) == list
        L = len(words)
        if L > 1:
            lst = []
            for i in range(L - 1):
                for k in range(1, skip + 2):
                    if i + k < L:
                        lst.append(join_string.join([words[i], words[i + k]]))
        else:
            # set it as unigram
            lst = NgramUtil.unigrams(words)
        return lst

    @staticmethod
    def trigrams(words, join_string, skip=0):
        """
           Input: a list of words, e.g., ["I", "am", "Denny"]
           Output: a list of trigram, e.g., ["I_am_Denny"]
        """
        assert type(words) == list
        L = len(words)
        if L > 2:
            lst = []
            for i in range(L - 2):
                for k1 in range(1, skip + 2):
                    for k2 in range(1, skip + 2):
                        if i + k1 < L and i + k1 + k2 < L:
                            lst.append(join_string.join([words[i], words[i + k1], words[i + k1 + k2]]))
        else:
            # set it as bigram
            lst = NgramUtil.bigrams(words, join_string, skip)
        return lst

    @staticmethod
    def fourgrams(words, join_string):
        """
            Input: a list of words, e.g., ["I", "am", "Denny", "boy"]
            Output: a list of trigram, e.g., ["I_am_Denny_boy"]
        """
        assert type(words) == list
        L = len(words)
        if L > 3:
            lst = []
            for i in xrange(L - 3):
                lst.append(join_string.join([words[i], words[i + 1], words[i + 2], words[i + 3]]))
        else:
            # set it as trigram
            lst = NgramUtil.trigrams(words, join_string)
        return lst

    @staticmethod
    def uniterms(words):
        return NgramUtil.unigrams(words)

    @staticmethod
    def biterms(words, join_string):
        """
            Input: a list of words, e.g., ["I", "am", "Denny", "boy"]
            Output: a list of biterm, e.g., ["I_am", "I_Denny", "I_boy", "am_Denny", "am_boy", "Denny_boy"]
        """
        assert type(words) == list
        L = len(words)
        if L > 1:
            lst = []
            for i in range(L - 1):
                for j in range(i + 1, L):
                    lst.append(join_string.join([words[i], words[j]]))
        else:
            # set it as uniterm
            lst = NgramUtil.uniterms(words)
        return lst

    @staticmethod
    def triterms(words, join_string):
        """
            Input: a list of words, e.g., ["I", "am", "Denny", "boy"]
            Output: a list of triterm, e.g., ["I_am_Denny", "I_am_boy", "I_Denny_boy", "am_Denny_boy"]
        """
        assert type(words) == list
        L = len(words)
        if L > 2:
            lst = []
            for i in xrange(L - 2):
                for j in xrange(i + 1, L - 1):
                    for k in xrange(j + 1, L):
                        lst.append(join_string.join([words[i], words[j], words[k]]))
        else:
            # set it as biterm
            lst = NgramUtil.biterms(words, join_string)
        return lst

    @staticmethod
    def fourterms(words, join_string):
        """
            Input: a list of words, e.g., ["I", "am", "Denny", "boy", "ha"]
            Output: a list of fourterm, e.g., ["I_am_Denny_boy", "I_am_Denny_ha", "I_am_boy_ha", "I_Denny_boy_ha", "am_Denny_boy_ha"]
        """
        assert type(words) == list
        L = len(words)
        if L > 3:
            lst = []
            for i in xrange(L - 3):
                for j in xrange(i + 1, L - 2):
                    for k in xrange(j + 1, L - 1):
                        for l in xrange(k + 1, L):
                            lst.append(join_string.join([words[i], words[j], words[k], words[l]]))
        else:
            # set it as triterm
            lst = NgramUtil.triterms(words, join_string)
        return lst

    @staticmethod
    def ngrams(words, ngram, join_string=" "):
        """
        wrapper for ngram
        """
        if ngram == 1:
            return NgramUtil.unigrams(words)
        elif ngram == 2:
            return NgramUtil.bigrams(words, join_string)
        elif ngram == 3:
            return NgramUtil.trigrams(words, join_string)
        elif ngram == 4:
            return NgramUtil.fourgrams(words, join_string)
        elif ngram == 12:
            unigram = NgramUtil.unigrams(words)
            bigram = [x for x in NgramUtil.bigrams(words, join_string) if len(x.split(join_string)) == 2]
            return unigram + bigram
        elif ngram == 123:
            unigram = NgramUtil.unigrams(words)
            bigram = [x for x in NgramUtil.bigrams(words, join_string) if len(x.split(join_string)) == 2]
            trigram = [x for x in NgramUtil.trigrams(words, join_string) if len(x.split(join_string)) == 3]
            return unigram + bigram + trigram

    @staticmethod
    def nterms(words, nterm, join_string=" "):
        """wrapper for nterm"""
        if nterm == 1:
            return NgramUtil.uniterms(words)
        elif nterm == 2:
            return NgramUtil.biterms(words, join_string)
        elif nterm == 3:
            return NgramUtil.triterms(words, join_string)
        elif nterm == 4:
            return NgramUtil.fourterms(words, join_string)

class DistanceUtil(object):
    """
    Tool of Distance
    """
    @staticmethod
    def dice_dist(A, B):
        if not isinstance(A, set):
            A = set(A)
        if not isinstance(B, set):
            B = set(B)
        return (2. * float(len(A.intersection(B)))) / (len(A) + len(B))
