import os
from timeit import default_timer as timer
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import nsml
import sentencepiece as spm
import pickle
from shutil import copyfile


import utils.data_utils as data_utils
from utils.utils import build_vocab
import data_local_loader_word
from feature.feature import Feature

from args.args_siam_lstm import get_config
from data_local_loader_word import get_dataloaders
#from model.charcnn_scorer import CharCNNScorer
from model.siamese_lstm import Siamese_LSTM


TRAIN_BATCH_IDX = 0


def bind_nsml(model, optimizer, config, sp, feature):
    def save(dir_name, *args, **kwargs):
        os.makedirs(dir_name, exist_ok=True)
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(state, os.path.join(dir_name, "model"))

        copyfile('vocab.model', os.path.join(dir_name, 'vocab.model'))
        with open(os.path.join(dir_name, 'feature.pickle'), 'wb') as fp:
            pickle.dump(feature, fp)

        print("saved")

    def load(dir_name, *args, **kwargs):
        state = torch.load(os.path.join(dir_name, "model"))
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        sp.load(os.path.join(dir_name, 'vocab.model'))
        feature.load_idf(os.path.join(dir_name, 'feature.pickle'))
        print("loaded")

    def infer(dataset_path):
        return _infer(model, config, dataset_path, sp, feature)

    nsml.bind(save=save, load=load, infer=infer)

def _infer(model, config, dataset_path, sp, feature):
    test_loader = data_local_loader_word.QuerySimDataLoader(
        dataset_path,
        "test_data",
        sp,
        feature,
        label_file_name=None,
        batch_size=config.batch_size,
        max_sequence_len=config.max_sequence_len,
        is_train=False,
        shuffle=False,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_logits = []
    for i, (uid, a_seqs, len_a_seqs, b_seqs, len_b_seqs, feats) in enumerate(test_loader):
        a_seqs, b_seqs, feats = a_seqs.to(device), b_seqs.to(device), feats.to(device)

        logits = model(a_seqs, b_seqs, feats, len_a_seqs, len_b_seqs)

        all_logits.append(torch.sigmoid(logits).data.cpu().numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    return all_logits


def run_epoch(
        epoch_idx,
        data_loader,
        model,
        criterion,
        optimizer,
        device,
        log_steps,
):
    total_loss = 0
    epoch_preds = []
    epoch_targets = []
    epoch_start = timer()

    for i, (uid, a_seqs, len_a_seqs, b_seqs, len_b_seqs, feats, labels) in enumerate(data_loader):
        a_seqs, b_seqs, labels, feats = a_seqs.to(device), b_seqs.to(device), labels.to(device), feats.to(device)

        logits = model(a_seqs, b_seqs, feats, len_a_seqs, len_b_seqs)

        loss = criterion(logits, labels)

        batch_loss = loss.data.cpu().item()
        total_loss += batch_loss

        if data_loader.is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global TRAIN_BATCH_IDX

            nsml.report(
                summary=False,
                step=TRAIN_BATCH_IDX,
                scope=locals(),
                **{
                    f"train__batch_loss": batch_loss,
                })

            if i > 0  and i % log_steps == 0:
                print(f"batch {i:5} loss > {loss.item():.4}")

            TRAIN_BATCH_IDX += 1

        epoch_preds.append(torch.sigmoid(logits).data.cpu().numpy())
        epoch_targets.append(labels.int().data.cpu().numpy())

    score = roc_auc_score(
        np.concatenate(epoch_targets, axis=0),
        np.concatenate(epoch_preds, axis=0),
    )

    mode = "train" if data_loader.is_train else "valid"
    print(f"epoch {epoch_idx:02} {mode} score > {score:.4} ({int(timer() - epoch_start)}s)")

    total_loss /= len(data_loader.dataset)
    return score, total_loss


if __name__ == "__main__":
    config = get_config()

    # Vocab load
    sp = spm.SentencePieceProcessor()
    feature = Feature()

    if config.mode == "train":
        data_dir = os.path.join(nsml.DATASET_PATH, "train", "train_data")
        build_vocab(os.path.join(data_dir, config.train_file_name))
        sp.load('vocab.model')

        feature.init_idf(os.path.join(data_dir, config.train_file_name))


    # random seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.random.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    model = Siamese_LSTM(
        vocab_size=2000,
        embed_size=config.char_embed_size,
        batch_size=config.batch_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        bidir=config.bidir,
        dropout=config.dropout,
        activation=config.activation,
        pad_ind=data_utils.PAD_IND,
        add_feat_size=feature.size
    ).to(device)

    print(str(model))

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
    )

    bind_nsml(model, optimizer, config, sp, feature)
    if config.pause:
        nsml.paused(scope=locals())

    if config.mode == "train":
        """
        nsml.load(checkpoint='3', session='team_6/18_tcls_query/231')
        nsml.save('ss')
        exit()
        """

        print("train")

        train_loader, valid_loader = get_dataloaders(config, sp, feature)
        criterion = nn.BCEWithLogitsLoss()

        num_batches = len(train_loader.dataset) // config.batch_size
        num_batches = num_batches + int((len(train_loader.dataset) % config.batch_size) > 0)
        print(f"number of batches per epoch: {num_batches}")

        best_epoch_idx = -1
        best_valid_score = 0
        early_stop_count = 0

        # train
        for epoch_idx in range(1, config.num_epochs + 1):

            def _run_epoch(data_loader):
                return run_epoch(
                    epoch_idx,
                    data_loader,
                    model,
                    criterion,
                    optimizer,
                    device,
                    config.log_steps
                )

            model.train()
            train_score, train_loss = _run_epoch(train_loader)

            # evaluate
            model.eval()
            with torch.no_grad():
                valid_score, valid_loss = _run_epoch(valid_loader)
                if best_valid_score < valid_score:
                    best_valid_score = valid_score
                    best_epoch_idx = epoch_idx
                    print(f"* best valid score {best_valid_score:.4} achieved at epoch {best_epoch_idx:02}")
                    early_stop_count = 0
                else:
                    early_stop_count += 1

                    if early_stop_count >= config.early_stop_threshold:
                        print("early stopping")
                        break

            nsml.report(
                summary=True,
                step=epoch_idx,
                scope=locals(),
                **{
                    "train__epoch_score": float(train_score),
                    "train__epoch_loss": float(train_loss),
                    "valid__epoch_score": float(valid_score),
                    "valid__epoch_loss": float(valid_loss),
                })

            nsml.save(str(epoch_idx))

    print(f"***** best valid score {best_valid_score:.4} achieved at epoch {best_epoch_idx:02}")
