
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import nsml
import random

from utils.data_utils import prepro_text
from utils.data_utils import text2ind_sp
from utils.data_utils import PAD_IND


class QuerySimDataset(Dataset):
    def __init__(
            self,
            data_dir,
            file_name,
            sp,
            feature,
            label_file_name=None,
            max_sequence_len=None,
            oversample=False
    ):
        self.data_file_name = file_name
        self.label_file_name = label_file_name
        self.data_dir = data_dir
        self.data_file_path = os.path.join(data_dir, file_name)
        self.sp = sp
        if label_file_name:
            self.label_file_path = os.path.join(data_dir, label_file_name)
        else:
            self.label_file_path = None

        self.max_sequence_len = max_sequence_len

        self.feature = feature
        self.oversample = oversample
        
        self._load_data(self.data_file_path, self.label_file_path)

    def _load_data(self, data_file_path, label_file_path):
        if label_file_path:
            with open(label_file_path) as f:
                labels = f.read().splitlines()
                labels = [int(label) for label in labels]

            self.labels = labels
        else:
            self.labels = None

        with open(data_file_path) as f:
            # data 읽어오는 부분
            data = f.read().splitlines()
            data = [line.split("\t") for line in data]

            if self.oversample:
                n_dup = labels.count(1)
                n_no_dup = labels.count(0)
                n_diff = abs(n_dup - n_no_dup)

                sample_label = 0 if n_dup > n_no_dup else 1
                sample_data = [d for (idx, d) in enumerate(data) if labels[idx] == sample_label]

                data += random.sample(sample_data, n_diff)
                labels += [sample_label] * n_diff


            _, a_seqs, b_seqs = list(zip(*data))

            # texts
            self.a_seqs = []
            self.b_seqs = []

            # features
            self.feats = []

            print("preprocessing data")

            for a_seq, b_seq in zip(a_seqs, b_seqs):
                self.a_seqs.append(prepro_text(a_seq))
                self.b_seqs.append(prepro_text(b_seq))

                self.feats.append(self.feature.extract_feature(a_seq, b_seq))

            # sequence dictionary
            seqs = sorted(list(set(self.a_seqs + self.b_seqs)))
            self.uid2seq = {uid: seq for uid, seq in enumerate(seqs)}
            self.uid2ind = {seq: uid for uid, seq in enumerate(seqs)}


    def __len__(self):
        assert len(self.a_seqs) == len(self.b_seqs)
        return len(self.a_seqs)

    def __getitem__(self, uid):
        a_seq = self.a_seqs[uid]
        b_seq = self.b_seqs[uid]

        a_seqs_idx = text2ind_sp(a_seq, self.max_sequence_len, self.sp)
        b_seqs_idx = text2ind_sp(b_seq, self.max_sequence_len, self.sp)

        if self.labels:
            label = self.labels[uid]
            return torch.tensor(uid), torch.tensor(a_seqs_idx), torch.tensor(b_seqs_idx), torch.tensor(label), torch.tensor(self.feats[uid])

        return torch.tensor(uid), torch.tensor(a_seqs_idx), torch.tensor(b_seqs_idx), torch.tensor(self.feats[uid])

def get_length(seq):
    return seq.tolist().index(2) + 1

def collate_fn(inputs):
    _inputs = list(zip(*inputs))

    if len(_inputs) == 5:
        uids, a_seqs, b_seqs, labels, feats = list(zip(*inputs))
    elif len(_inputs) == 4:
        uids, a_seqs, b_seqs, feats = list(zip(*inputs))
        labels = None
    else:
        raise Exception("Invalid inputs")

    len_a_seqs = torch.tensor([get_length(a_seq) for a_seq in a_seqs])
    len_b_seqs = torch.tensor([get_length(b_seq) for b_seq in b_seqs])

    seqs = nn.utils.rnn.pad_sequence(a_seqs + b_seqs, batch_first=True, padding_value=PAD_IND)
    a_seqs, b_seqs = torch.split(seqs, len(inputs), dim=0)

    # tuple of tensors -> tensor of 2d array
    feats = nn.utils.rnn.pad_sequence(feats, batch_first=True, padding_value=PAD_IND)

    batch = [
        torch.stack(uids, dim=0),
        a_seqs,
        len_a_seqs,
        b_seqs,
        len_b_seqs,
        feats,
    ]
    if labels:
        batch.append(torch.stack(labels, dim=0).float())

    return batch


class QuerySimDataLoader(DataLoader):
    def __init__(
            self,
            data_dir,
            file_name,
            sp,
            feature,
            label_file_name=None,
            batch_size=64,
            max_sequence_len=128,
            is_train=False,
            shuffle=False,
            drop_last = False,
            oversample=False
    ):
        super(QuerySimDataLoader, self).__init__(
            QuerySimDataset(
                data_dir,
                file_name,
                sp,
                feature,
                label_file_name=label_file_name,
                max_sequence_len=max_sequence_len,
                oversample=oversample
            ),
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last
        )

        self.is_train = is_train


def get_dataloaders(config, sp, feature):
    data_dir = os.path.join(nsml.DATASET_PATH, "train", "train_data")

    train_loader = QuerySimDataLoader(
        data_dir,
        config.train_file_name,
        sp,
        feature,
        config.train_label_file_name,
        batch_size=config.batch_size,
        max_sequence_len=config.max_sequence_len,
        is_train=True,
        shuffle=True,
        drop_last=False,
        oversample=False
    )
    valid_loader = QuerySimDataLoader(
        data_dir,
        config.valid_file_name,
        sp,
        feature,
        config.valid_label_file_name,
        batch_size=config.batch_size,
        max_sequence_len=config.max_sequence_len,
        is_train=False,
        shuffle=False,
    )

    return train_loader, valid_loader
