import os
from .data_utils import prepro_text
import sentencepiece as spm


# Tokenizing 된 애들로 vocab 만들기
def build_vocab(data_file_path):
    with open(data_file_path) as f:
        # data 읽어오는 부분
        data = f.read().splitlines()
        data = [line.split("\t") for line in data]

        _, a_seqs, b_seqs = list(zip(*data))

        with open('corpus.txt', 'w') as wf:
            for a_seq, b_seq in zip(a_seqs, b_seqs):
                wf.write(prepro_text(a_seq) + '\n')
                wf.write(prepro_text(b_seq) + '\n')

        spm.SentencePieceTrainer.train('--input=corpus.txt --model_prefix=vocab --vocab_size=2000')


# Vocab 을 이용해 indexing -> Embedding
