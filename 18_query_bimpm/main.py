import argparse
import os
import numpy as np
import torch
import nsml
import random

from nsml import DATASET_PATH, IS_ON_NSML
from torch.utils.data import DataLoader

from data import Vocabulary, Feature, Dataset
from trainer import Trainer


def inference(path, model, vocab, feature, config, **kwargs):
    model.eval()
    test_question_file_path = os.path.join(path, config.test_file_name)
    test_dataset = Dataset(test_question_file_path, None, vocab, feature, mode='test')
    test_data_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    predict_results = []
    for step, (ebd1, ebd2, f) in enumerate(test_data_loader):
        ebd1, ebd2 = ebd1.to(config.device), ebd2.to(config.device)
        f = f.to(config.device)
        outputs = model(ebd1, ebd2, f, is_train=False)

        predict_results.append(torch.sigmoid(outputs).data.cpu().numpy())

    predict_results = np.concatenate(predict_results, axis=0)
    return predict_results


def bind_model(model, vocab, feature, config, **kwargs):
    def save(path, *args, **kwargs):
        # save the model with 'state' dictionary.
        state = {
            'model': model.state_dict(),
        }
        torch.save(state, os.path.join(path, 'model'))

        vocab.save_vocab(path)
        feature.save_data(path)

    def load(path, *args, **kwargs):
        state = torch.load(os.path.join(path, 'model'))
        model.load_state_dict(state['model'])

        vocab.load_vocab(path)
        feature.load_data(path)

    def infer(path, **kwargs):
        return inference(path, model, vocab, feature, config)

    nsml.bind(save, load, infer)


def main(config, local):
    # random seed
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    torch.random.manual_seed(config.random_seed)
    if config.device == 'cuda':
        torch.cuda.manual_seed_all(config.random_seed)

    vocab = Vocabulary(config)
    print(f'Vocabulary loaded')
    feature = Feature(config)
    print(f'Feature data loaded')

    setattr(config, 'char_vocab_size', 0)
    setattr(config, 'class_size', 1)

    if config.mode == 'train':
        train_question_file_path = os.path.join(config.data_dir, config.train_file_name)
        train_label_file_path = os.path.join(config.data_dir, config.train_label_file_name)
        train_dataset = Dataset(train_question_file_path, train_label_file_path,
                                vocab, feature, mode='train')
        train_data_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

        validation_question_file_path = os.path.join(config.data_dir, config.validation_file_name)
        validation_label_file_path = os.path.join(config.data_dir, config.validation_label_file_name)
        validation_dataset = Dataset(validation_question_file_path, validation_label_file_path,
                                     vocab, feature, mode='validation')
        validation_data_loader = DataLoader(validation_dataset, batch_size=config.batch_size)
    else:
        train_data_loader = None
        validation_data_loader = None
    print(f'{config.mode} Dataset loaded')

    trainer = Trainer(config, feature, train_data_loader, validation_data_loader)
    print(f'Trainer loaded')

    if nsml.IS_ON_NSML:
        bind_model(trainer.model, vocab, feature, config)

        if config.pause:
            nsml.paused(scope=local)

    if config.mode == 'train':
        print(f'Starting training')
        trainer.train()
        print(f'Finishing training')


if __name__ == '__main__':
    print(f'Starting')
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--train_file_name', type=str, default='train_data')
    parser.add_argument('--train_label_file_name', type=str, default='train_label')
    parser.add_argument('--validation_file_name', type=str, default='valid_data')
    parser.add_argument('--validation_label_file_name', type=str, default='valid_label')
    parser.add_argument('--test_file_name', type=str, default='test_data')

    # Vocabulary
    parser.add_argument('--vocab_file_prefix', type=str, default='vocab')
    parser.add_argument('--vocab_size', type=int, default=1500)
    parser.add_argument('--max_sequence_len', type=int, default=128)

    # Feature
    parser.add_argument('--feature_data_file_prefix', type=str, default='feature_data')
    parser.add_argument('--ngram_max', type=int, default=3)
    parser.add_argument('--nterm_max', type=int, default=2)

    # BIMPM
    parser.add_argument('--word-dim', default=300, type=int)
    parser.add_argument('--use-char-emb', default=False, action='store_true')
    parser.add_argument('--char-dim', default=20, type=int)
    parser.add_argument('--char-hidden-size', default=50, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--hidden-size', default=100, type=int)
    parser.add_argument('--num-perspective', default=20, type=int)

    # Training setting
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--early_stop_threshold', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    # NSML
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--iteration', type=str, default='0')
    parser.add_argument('--pause', type=int, default=0)

    # Etc
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--validation_interval', type=int, default=100)

    config = parser.parse_args()

    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if IS_ON_NSML:
        config.data_dir = os.path.join(DATASET_PATH, 'train', 'train_data')

    print(f'Arguments processed')
    main(config, locals())

