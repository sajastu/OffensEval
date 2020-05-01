from os.path import join as pjoin

import pandas as pd
import torch
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, RandomSampler, DataLoader
from tqdm import tqdm
from transformers import BertTokenizer


def label_switch(old_label):
    if True:
        if old_label == 'OFF':
            return 1
        elif old_label == 'NOT':
            return 0
        if old_label == 'TIN':
            return 1
        elif old_label == 'UNT':
            return 0
        if old_label == 'IND':
            return 0
        elif old_label == 'GRP':
            return 1
        elif old_label == 'OTH':
            return 2


def attention_mask(ids):
    masks = list()
    for sent in ids:
        mask = [int(token_id > 0) for token_id in sent]
        masks.append(mask)
    return masks


class MyDataLoader(object):
    def __init__(self, args):
        self.data_base_dir = args.data_dir
        self.seed = args.seed
        self.max_len = args.max_len
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.args = args

    def load_train_set(self, task):
        set_path = pjoin(self.data_base_dir, 'olid-training-v1.0.tsv')
        if task == 'a':
            OLID = pd.read_csv(set_path, sep='\t', header=0)[['id', 'tweet', 'subtask_a']]
        elif task == 'b':
            OLID = pd.read_csv(set_path, sep='\t', header=0)[['id', 'tweet', 'subtask_b']]
            OLID = OLID[~pd.isna(OLID["subtask_b"])]
        elif task == 'c':
            OLID = pd.read_csv(set_path, sep='\t', header=0)[['id', 'tweet', 'subtask_c']]
            OLID = OLID[~pd.isna(OLID["subtask_c"])]
        OLID.columns = ['id', 'text', 'label']
        OLID = OLID.assign(Platform='Twitter')
        tqdm.pandas(desc='Converting labels...')
        OLID['label'] = OLID['label'].progress_apply(label_switch)
        OLID = OLID.sample(frac=1, random_state=self.seed)
        OLID.reset_index(drop=True, inplace=True)
        self.train_set = OLID

    def load_dev_set(self, task='a', data='testset-levela.tsv', label='labels-levela.csv', random_seed=233):
        data_file = pjoin(self.data_base_dir, data)
        label_file = pjoin(self.data_base_dir, label)
        if task == 'b':
            data_file = data_file.replace('levela', 'levelb')
            if label:
                label_file = label_file.replace('levela', 'levelb')
        if task == 'c':
            data_file = data_file.replace('levela', 'levelc')
            if label:
                label_file = label_file.replace('levela', 'levelc')

        test_data = pd.read_csv(data_file, sep='\t', header=0)[['id', 'tweet']]
        test_data.columns = ['id', 'text']

        if label_file:
            test_label = pd.read_csv(label_file, sep=',', header=None)
            test_label.columns = ['id', 'label']
            tqdm.pandas(desc="Converting labels...")
            test_label['label'] = test_label['label'].progress_apply(label_switch)
            OLID_dev = pd.concat([test_label[['id']], test_data[['text']], test_label[['label']]],
                                 axis=1)
        else:
            OLID_dev = pd.concat([test_data[['id']], test_data[['text']]], axis=1)

        OLID_dev = OLID_dev.assign(Platform='Twitter')
        OLID_dev = OLID_dev.sample(frac=1, random_state=random_seed)
        OLID_dev.reset_index(drop=True, inplace=True)
        self.dev_set = OLID_dev

    def load_test_set(self, task='a', data='testset_a_2020.tsv', label='englishA-goldlabels.csv', random_seed=233):
        data_file = pjoin(self.data_base_dir, data)
        label_file = pjoin(self.data_base_dir, label)
        if task == 'b':
            data_file = data_file.replace('A', 'B')
            if label:
                label_file = label_file.replace('A', 'B')
        if task == 'c':
            data_file = data_file.replace('A', 'C')
            if label:
                label_file = label_file.replace('a', 'c')

        test_data = pd.read_csv(data_file, sep='\t', header=0)[['id', 'tweet']]
        test_data.columns = ['id', 'text']

        if label_file:
            test_label = pd.read_csv(label_file, sep=',', header=None)
            test_label.columns = ['id', 'label']
            tqdm.pandas(desc="Converting labels...")
            test_label['label'] = test_label['label'].progress_apply(label_switch)
            OLID_test = pd.concat([test_label[['id']], test_data[['text']], test_label[['label']]],
                                  axis=1)
        else:
            OLID_test = pd.concat([test_data[['id']], test_data[['text']]], axis=1)

        OLID_test = OLID_test.assign(Platform='Twitter')
        OLID_test = OLID_test.sample(frac=1, random_state=random_seed)
        OLID_test.reset_index(drop=True, inplace=True)
        import pdb;pdb
        self.test_set = OLID_test

    def sent_label_extractor(self, is_eval=False, is_test=False):
        set = list()
        if is_eval:
            set = self.dev_set
        elif is_test:
            set = self.test_set
        else:
            set = self.train_set
        sents = set["text"].tolist()
        labels = set["label"].tolist()
        if is_test or is_eval:
            tweetids = set['id'].tolist()
            return sents, labels, tweetids
        else:
            return sents, labels

    def prepare_bert_data(self, sents, labels):
        ids = self._encode_sent_to_ids_bert(sents)
        return self._encode_to_features(ids, labels, self.max_len)

    def _encode_sent_to_ids_bert(self, sents):
        ids = []
        for sent in sents:
            encoded_sent = self.bert_tokenizer.encode(sent, add_special_tokens=True)
            ids.append(encoded_sent)
        return ids

    def _encode_to_features(self, ids, labels, maxlen):
        ids = pad_sequences(ids, maxlen=maxlen, dtype='long', value=0, truncating='post', padding='post')
        masks = attention_mask(ids)
        inputs = torch.tensor(ids).to(torch.int64)
        labels = torch.tensor(labels).to(torch.int64)
        masks = torch.tensor(masks)
        return inputs, masks, labels

    def get_dataloader(self, inputs, masks, labels):
        data = TensorDataset(inputs, masks, labels)
        sampler = RandomSampler(data)
        return DataLoader(data, sampler=sampler, batch_size=self.args.batch_size)
