import nsml
import numpy as np
from timeit import default_timer
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from time import gmtime, strftime
from random import *

from model.bimpm import BIMPM


class Trainer:
    global_step = 0

    def __init__(self, config, feature, train_data_loader=None, validation_data_loader=None):
        self.config = config
        self.train_data_loader = train_data_loader
        self.validation_data_loader = validation_data_loader

        self.model = BIMPM(config, feature).to(config.device)

        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.config.learning_rate)
        self.criterion = nn.BCEWithLogitsLoss()

    def run_epoch(self, epoch, data_loader, mode):
        total_loss = 0
        epoch_predicts = []
        epoch_labels = []
        epoch_start = default_timer()

        for step, batch in enumerate(data_loader):
            batch = tuple(t.to(self.config.device) for t in batch)
            embeddings1, embeddings2, features, labels = batch

            outputs = self.model(embeddings1, embeddings2, features)
            loss = self.criterion(outputs, labels.float())

            batch_loss = loss.data.cpu().item()
            total_loss += batch_loss

            if mode == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                nsml.report(summary=False, step=self.global_step, scope=locals(), train__batch_loss=batch_loss)

                self.global_step += 1

            epoch_predicts.append(torch.sigmoid(outputs).data.cpu().numpy())
            epoch_labels.append(labels.int().data.cpu().numpy())

        score = roc_auc_score(np.concatenate(epoch_labels, axis=0), np.concatenate(epoch_predicts, axis=0))
        time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        print(f'{time}: epoch {epoch:02} {mode} score = {score:.4} ({int(default_timer() - epoch_start)}s)')

        return score, total_loss / len(data_loader.dataset)

    def train(self):
        best_epoch = None
        best_validation_score = 0
        early_stop_count = 0

        for epoch in range(self.config.num_epochs):
            # training
            self.model.train()
            train_score, train_loss = self.run_epoch(epoch, self.train_data_loader, 'train')

            # validation
            self.model.eval()
            with torch.no_grad():
                validation_score, validation_loss = self.run_epoch(epoch, self.validation_data_loader, 'validation')
                if best_validation_score < validation_score:
                    best_validation_score = validation_score
                    best_epoch = epoch
                    print(f"* best validation score {best_validation_score:.4} achieved at epoch {best_epoch:02}")
                    early_stop_count = 0
                else:
                    early_stop_count += 1
                    if early_stop_count >= self.config.early_stop_threshold:
                        print("early stopping")
                        break

            nsml.report(
                summary=True, step=epoch, scope=locals(),
                train__epoch_score=float(train_score), train__epoch_loss=float(train_loss),
                validation__epoch_score=float(validation_score), validation__epoch_loss=float(validation_loss))

            nsml.save(str(epoch))

        print(f"***** best validation score {best_validation_score:.4} achieved at epoch {best_epoch:02}")