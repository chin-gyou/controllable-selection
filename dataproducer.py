from torch.utils.data import Dataset, DataLoader
import numpy as np
from config import *
import pickle

"""
data_dir: directory containing all preprocessed texts
"""
class tabledata(Dataset):
    def __init__(self, data_dir, mode='train'):
        print('initializing', mode, 'data')
        data_path = [data_dir + '/'+mode+'/title.30k.id_end', data_dir + '/'+mode+'/article.30k.id', data_dir + '/'+mode+'/embed.30k.mask']
        #data_path = ['outputs.id', data_dir + '/'+mode+'/article.bert.id']
        #data_path = ['decoding/xsum/test2.id', data_dir + '/'+mode+'/article.bert.id']
        #data_path = ['xsumcopy.id', data_dir + '/'+mode+'/article.n30k.id']
        datalists = [id2list(p) for p in data_path]
        #textlists = text2list(text_path[0], text_path[1])
        self.data = list(zip(*(datalists)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

"""
parse a file into a list of id tokens
each line is a sample, splited by ' '
"""
def id2list(path):
    #print(path)
    ids = open(path, 'r', encoding = 'utf8').read().strip().split('\n')
    #m = []
    #try:
    ids = [list(map(int, text.strip().split(' '))) for text in ids]
    
    #if path == 'xsumcopy.id':
    #    ids = [[i+1 for i in text]for text in ids]

    #for text in ids:
        #    m.append(list(map(int, text.strip().split(' '))))
    #except:
    #    print(text)
    return ids
    #return m

def text2list(spath, epath):
    stexts = open(spath, 'r', encoding = 'utf8').read().strip().split('\n')
    etexts = open(epath, 'r', encoding = 'utf8').read().strip().split('\n')
    stexts = [text.strip().split(' ') for text in stexts]
    etexts = [text.strip().split(' ') for text in etexts]
    return [etexts, stexts]

#create dataset classes for train, valid and test data
def create_datasets(data_dir, bsize):
    modes = ('train', 'valid', 'test')
    shuffleds = (True, False, False)
    b_sizes = (bsize, 2*bsize, 2*bsize)
    return [DataLoader(tabledata(data_dir, mode), batch_size = b_size, shuffle=shuffled, collate_fn = merge_sample) for mode, b_size, shuffled in zip(modes, b_sizes, shuffleds)]

#padd sequences to some max_len and convert to tensor
def padd(l, max_len, max_limit):
    ml = min(max_len, max_limit)
    if l[0] is list:
        nl = [top[:ml] if len(top) > ml else top for top in l]
        return torch.tensor(nl).transpose(0, 1).to(DEVICE)
    nl = [text+[PAD]*(ml-len(text)) if ml>len(text) else text[:ml] for text in l]

    return torch.tensor(nl).t().to(DEVICE)

def merge_sample(batch):
    dicted_data = [[b[i] for b in batch] for i in range(3)]
    max_lens = [max([len(text) for text in v]) for v in dicted_data]
    max_limit = [dec_limit, enc_limit]
    #max_lens = max_limit

    #padd data up to the maximum length
    dicted_data = [padd(v, le, li) for v, li, le in zip(dicted_data, max_limit, max_lens)]
    return dicted_data

if __name__ == '__main__':
    train_data = tabledata('../xsum', 'test')
    dataloader = DataLoader(train_data,batch_size=3,shuffle=False, collate_fn = merge_sample)
    for i, batch in enumerate(dataloader):
        print(i)
        if i == 0:
            data = batch
            print(len(data))
            print('0: ', data[0].size())
            print('1: ', data[1].size())
            print('2: ', data[2].size())
            break
