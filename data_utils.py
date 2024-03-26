import os
import json
import torch
from functools import partial
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):

    def __init__(self, raw_data, label_dict, tokenizer, model_name, method):
        label_list = list(label_dict.keys()) if method not in ['ce', 'scl'] else []
        sep_token = ['[SEP]'] if model_name == 'bert' else ['</s>']
        dataset = list()
        for data in raw_data:
            tokens = data['text'].lower().split(' ')
            label_id = label_dict[data['label']]
            dataset.append((label_list + sep_token + tokens, label_id))
        self._dataset = dataset

    def __getitem__(self, index):
        return self._dataset[index]

    def __len__(self):
        return len(self._dataset)


def my_collate(batch, tokenizer, method, num_classes):
    tokens, label_ids = map(list, zip(*batch))
    text_ids = tokenizer(tokens,
                         padding=True,
                         truncation=True,
                         max_length=256,
                         is_split_into_words=True,
                         add_special_tokens=True,
                         return_tensors='pt')
    if method not in ['ce', 'scl']:
        positions = torch.zeros_like(text_ids['input_ids'])
        positions[:, num_classes:] = torch.arange(0, text_ids['input_ids'].size(1)-num_classes)
        text_ids['position_ids'] = positions
    return text_ids, torch.tensor(label_ids)


def load_data(dataset, data_dir, tokenizer, train_batch_size, test_batch_size, model_name, method, workers):
    if dataset == 'amazon-jewel-debug':
        train_data = json.load(open(os.path.join(data_dir, 'Clothing_Shoes_and_Jewelry_debug_train_debug.json'), 'r', encoding='utf-8'))
        val_data = json.load(open(os.path.join(data_dir, 'Clothing_Shoes_and_Jewelry_debug_val_debug.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'Clothing_Shoes_and_Jewelry_debug_test_debug.json'), 'r', encoding='utf-8'))
        label_dict = {'positive': 0, 'negative': 1}
    elif dataset == 'amazon-jewel':
        train_data = json.load(open(os.path.join(data_dir, 'Clothing_Shoes_and_Jewelry_train.json'), 'r', encoding='utf-8'))
        val_data = json.load(open(os.path.join(data_dir, 'Clothing_Shoes_and_Jewelry_val.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'Clothing_Shoes_and_Jewelry_test.json'), 'r', encoding='utf-8'))
        label_dict = {'positive': 0, 'negative': 1}
    elif dataset == 'amazon-jewel-5-debug':
        train_data = json.load(open(os.path.join(data_dir, 'amazon-jewel-five-debug_train.json'), 'r', encoding='utf-8'))
        val_data = json.load(open(os.path.join(data_dir, 'amazon-jewel-five-debug_val.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'amazon-jewel-five-debug_test.json'), 'r', encoding='utf-8'))
        label_dict = {'positive': 0, 'negative': 1, 'neutral': 2, 'very positive': 3, 'very negative': 4}

    else:
        raise ValueError('unknown dataset')
    trainset = MyDataset(train_data, label_dict, tokenizer, model_name, method)
    valset = MyDataset(val_data, label_dict, tokenizer, model_name, method)
    testset = MyDataset(test_data, label_dict, tokenizer, model_name, method)
    collate_fn = partial(my_collate, tokenizer=tokenizer, method=method, num_classes=len(label_dict))
    train_dataloader = DataLoader(trainset, train_batch_size, shuffle=True, num_workers=workers, collate_fn=collate_fn, pin_memory=True)
    test_dataloader = DataLoader(testset, test_batch_size, shuffle=False, num_workers=workers, collate_fn=collate_fn, pin_memory=True)
    val_dataloader = DataLoader(valset, test_batch_size, shuffle=False, num_workers=workers, collate_fn=collate_fn, pin_memory=True)
    return train_dataloader, val_dataloader,test_dataloader
