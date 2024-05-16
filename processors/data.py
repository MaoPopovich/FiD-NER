# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''
consider each example of my data is a tuple composed with three parts
original: original sequence, which is a list of string
related: related passages, which is a list of string
label: label list, which is a list of BIO label(string) len(label)==label(original)
'''
import torch
import random
import json
import numpy as np

class NerDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 n_context=None):
        self.data = data
        self.n_context = n_context

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        original = example['original']
        labels = []
        for x in example['labels']:
            if 'M-' in x:
                labels.append(x.replace('M-','I-'))
            elif 'E-' in x:
                labels.append(x.replace('E-', 'I-'))
            else:
                labels.append(x)

        if 'similar' in example and self.n_context is not None:
            len_similar = len(example['similar'])
            passages = []
            if len_similar >= self.n_context:
                passages = example['similar'][:self.n_context]
            else:
                tmp = [example['similar'][0] for i in range(self.n_context-len_similar)]
                passages = example['similar']
                passages.extend(tmp)
        else:
            passages = None


        return {
            'original' : original,
            'passages' : passages,
            'labels' : labels
            
        }

def load_data(data_dir=None, data_type='train'):
    data_path = data_dir + data_type + '_split.json'
    assert data_path
    if data_path.endswith('.json'):
        with open(data_path, 'r') as fin:
            examples = json.load(fin)

    return examples
    
    

def encode_passages(batch_text_passages, tokenizer, max_seq_length):
    passage_ids, passage_masks, passage_segment_ids = [], [], []
    for k, text_passages in enumerate(batch_text_passages):
        p = tokenizer.batch_encode_plus(
            text_passages,
            max_length=max_seq_length,
            padding='max_length',
            return_tensors='pt',
            truncation=True
        )
        passage_ids.append(p['input_ids'][None])
        passage_masks.append(p['attention_mask'][None])
        passage_segment_ids.append(p['token_type_ids'][None])

    passage_ids = torch.cat(passage_ids, dim=0)
    passage_masks = torch.cat(passage_masks, dim=0)
    passage_segment_ids = torch.cat(passage_segment_ids,dim=0)
    return passage_ids, passage_masks, passage_segment_ids

class Collator(object):
    def __init__(self, max_seq_length, tokenizer,label_list):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.label_list = label_list
        self.label_map = {label: i for i, label in enumerate(self.label_list)}
    
    def input_proc(self,text,labels):
        tokens = self.tokenizer.tokenize(text)
        label_id = [self.label_map[x] for x in labels]
        special_tokens_count = 2
        if len(label_id) > self.max_seq_length - special_tokens_count:
            tokens = tokens[: (self.max_seq_length - special_tokens_count)]
            label_id = label_id[: (self.max_seq_length - special_tokens_count)]
        
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        label_id = [self.label_map['O']] + label_id + [self.label_map['O']]
        input_segment_id = [1]+ [0] * (len(tokens)-1)

        input_id = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_id)

        # Zero-pad up to the sequence length.
        padding_length = self.max_seq_length - len(input_id)
        input_id += [0] * padding_length
        input_mask += [0] * padding_length
        input_segment_id += [0] * padding_length
        label_id += [0] * padding_length

        assert len(input_id) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(input_segment_id) == self.max_seq_length
        if len(label_id) != self.max_seq_length:
            print(text)
            print(len(input_id))
            print(len(label_id))
        assert len(label_id) == self.max_seq_length

        return (input_id, input_mask, input_segment_id, label_id)

    def __call__(self, batch):
        assert(batch[0]['original'] != None)
        input_ids = []
        input_masks = []
        input_segment_ids = []
        labels = []
        passages_batch = []
        for example in batch:
            if isinstance(example['original'],list):
                example['original'] = " ".join(example['original'])
            example['passages'].insert(0,example['original']) 
            passages_batch.append(example['passages'])
            # input_id, input_mask, input_segment_id, label_id = self.input_proc(example['original'],example['labels'])
            _, _, _, label_id = self.input_proc(example['original'],example['labels']) 
            labels.append(label_id)
            # input_ids.append(input_id)
            # input_masks.append(input_mask)
            # input_segment_ids.append(input_segment_id)

        # input_ids = torch.tensor(input_ids,dtype=torch.long)
        # input_masks = torch.tensor(input_masks,dtype=torch.long)
        # input_segment_ids = torch.tensor(input_segment_ids,dtype=torch.long)
        labels = torch.tensor(labels,dtype=torch.long)
        passage_ids, passage_masks, passage_segment_ids = encode_passages(passages_batch,
                                                     self.tokenizer,
                                                     self.max_seq_length)

        # return (input_ids, input_masks, input_segment_ids, labels, passage_ids, passage_masks, passage_segment_ids)
        return (passage_ids, passage_masks, passage_segment_ids, labels)



