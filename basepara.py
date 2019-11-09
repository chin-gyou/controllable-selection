import torch.nn as nn
import numpy as np

from dataproducer import *
#from dataproducer import *
from helper import *
import pickle
import os

#initialize hidden states
def init_state(b_size, h_size, device):
    return torch.zeros(b_size, h_size).to(device)

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, drop_out = 0.3, bi = False, num_layers = 1):
        super().__init__()
        #parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bi = bi
        self.do = nn.Dropout(p=drop_out)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, dropout=drop_out if num_layers > 1 else 0., bidirectional=bi)

    """
    return the hidden states after processing a sequence
    seq: max_len*batch_size*input_size
    slen: batch_size, length of sequence
    return hidden: max_len*[batch_size*h_size]
    """
    def run(self, seq, slen, mode = 0):
        slen, perm_idx = slen.sort(0, descending=True)
        seq = seq[:, perm_idx, :]
        seq = self.do(seq) if mode == 0 else seq
        packed_input = nn.utils.rnn.pack_padded_sequence(seq, slen.data.tolist())
        packed_output, (ht, ct) = self.lstm(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output)
        
        _, unperm_idx = perm_idx.sort(0)
        output = output[:,unperm_idx,:]
        ht = ht[:, unperm_idx, :]
        return output, ht

    def forward(self, inputs, hidden, cell, mode=0):
        i = inputs
        #if training, use drop out
        if mode==0:
            i = self.do(i)
        _, (h, c) = self.lstm(i, (hidden, cell))
        return h,c
"""
input_size: size of context except word embedding
embeddings: vocab_size*embed_size, type nn.Embedding
context_size: size of context vector, must not be 0 when concated=True
concated: whether concat context vector at each input step
"""
class DecoderRNN(nn.Module):
    def __init__(self, embeddings, hidden_size, context_size=0, drop_out = 0.3, i_size = 0):
        super().__init__()
        #parameters
        i_size = context_size if i_size == 0 else i_size
        self.vocab_size, self.embed_size = embeddings.weight.size()
        self.embedding = embeddings
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.cell = nn.LSTMCell(self.embed_size + context_size, hidden_size)
        self.decode1 = nn.Linear(hidden_size + i_size, self.hidden_size)
        self.decode2 = nn.Linear(self.hidden_size, self.vocab_size)
        if i_size >0:
            self.initS = nn.Linear(i_size, hidden_size)
            print(self.initS)
            self.initC = nn.Linear(i_size, hidden_size)
        self.do = nn.Dropout(p=drop_out)

    """
    input_word: batch_size
    context: batch_size*context_size
    initial hidden: 0
    mask: EOT state of current word
    """
    def forward(self, input_word, hidden,cellstat, context=None, mode=0):
        if context is not None:
            inputs = torch.cat((context, self.embedding(input_word)), 1)
        else:
            inputs = self.embedding(input_word)
        if mode==0:
            inputs = self.do(inputs)
        new_h, new_c = self.cell(inputs, (hidden, cellstat))#batch_size*hidden_size
        return new_h, new_c

    def decode(self, h, c):
        output1 = nonlinear(self.decode1(torch.cat((h,c),1)))
        output2 = self.decode2(output1)#batch_size*vocab_size
        return output2
    """
    return the output list after decoding a sequence
    seq: max_len*batch_size
    context: batch_size*context_size
    outputs: max_len*[batch_size*vocab_size]
    """
    def run(self, seq, context=None, mode = 0):
        max_len, batch_size = seq.size()
        d_hidden = nonlinear(self.initS(context)) if context is not None else init_state(batch_size, self.hidden_size)
        d_cell = nonlinear(self.initC(context)) if context is not None else init_state(batch_size, self.hidden_size)
        outputs = []
        for i in range(max_len):
            d_hidden, d_cell, output = self.forward(seq[i], d_hidden, d_cell, context, mode)
            outputs.append(output)
        return outputs

class base(nn.Module):
    def __init__(self):
        super().__init__()
        self.step = torch.tensor(0, dtype = torch.int, requires_grad=False)
    
    """
    cross entropy loss between tagets and outputs
    targets: max_len*batch_size
    outputs: max_len*batch_size*vocab_size
    """
    def c_ent(self, targets, outputs, ignore = True):
        t_len=torch.sum((targets!=PAD).float()) if ignore else torch.sum((targets > -1).float())
        loss = nn.CrossEntropyLoss(size_average=False, ignore_index=PAD) if ignore else nn.CrossEntropyLoss(size_average=False)
        t_loss = 0
        for i in range(targets.size(0)):
            t_loss += loss(outputs[i], targets[i])
        return t_loss/t_len

    def optimize(self):
        torch.nn.utils.clip_grad_norm_(self.parameters(), MAX_CLIP)
        self.optim.step()
        self.optim.zero_grad()
        self.step+=1
  
    def output_decode(self, dataloader,w_file, tokenizer, attn=False):
        self.mode = 2
        outputs, oids = [], []
        if attn:
            mfs = []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                print('batch',i)
                #if i > 20:
                #    break
                if attn:
                    dec_out, mf = self.decode(batch)
                else:
                    dec_out = self.decode(batch)#b_size*dec_len
                ids = [(dec_out[i]).tolist() for i in range(dec_out.size(0))]
                ids = [l[:l.index(END)] if END in l else l for l in ids]
                #sents = [' '.join(sp.DecodeIds(l).replace('\u2047','').split())+'\n' for l in ids]
                sids = [' '.join([str(i) for i in l]) + '\n' for l in ids]
                sents = [' '.join(tokenizer.convert_ids_to_tokens(l))+'\n' for l in ids]
                #sents = [' '.join(sp.DecodeIds(l).split())+'\n' for l in ids]
                if attn:
                    mfs.extend([' '.join([str(j) for j in mf[i].tolist()])+'\n' for i in range(mf.size(0))])
                outputs.extend(sents)
                oids.extend(sids)
        with open(w_file, 'w', encoding='utf8') as f:
            f.writelines(outputs)
        with open(w_file + '.id', 'w', encoding='utf8') as f:
            f.writelines(oids)
        if attn:
            with open(w_file+'_mf', 'w', encoding='utf8') as f:
                f.writelines(mfs)
    """
    validate process
    dataloader: dataloader for the validation data
    w_file: written information about the validation
    """
    def validate(self, dataloader, w_file):
        t_loss = 0
        o_loss = 0
        e_loss = 0
        r_loss = 0
        self.mode = 1
        net = self#torch.nn.DataParallel(self, device_ids=range(N_GPU), dim = 1)
        #net.to(DEVICE)
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                #if i < 6333:
                #    continue
                #if i > 3:
                #    break
                forwarded=net(batch)
                c = self.cost(forwarded)
                t_loss += c[0]
                e_loss += c[1]
                r_loss += c[2]
                o_loss += c[3]
                print('[Validation]Mini-Batches run : %d\tTLoss: %.2f\tELoss: %.2f\tRLoss: %.2f\tOLoss: %.2f' % (i+1, e_loss / r_loss, e_loss/ (i+1), r_loss/(i+1), o_loss/(i+1)))
        print('Final loss : %f' % (e_loss/r_loss))
        with open(w_file, 'a') as f:
            f.write('Loss : %f DLoss: %f RLoss: %f OLoss: %f\n' % (e_loss/r_loss, e_loss/len(dataloader), r_loss/len(dataloader), o_loss/len(dataloader)))
        return (e_loss)/r_loss
    
    def gen_top(self, dataloader, w_file):
        tops = []
        self.mode = 1
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                print(i)
                forwarded=self(batch)
                tops.extend(forwarded)
        print(len(tops))
        with open(w_file, 'wb') as f:
           pickle.dump(tops, f) 

    """
    train process
    dataloaderL dataloader for the training data
    """
    def train(self, dataloader, lazy_step = 1):
        self.mode = 0
        net = self#torch.nn.DataParallel(self, device_ids=range(N_GPU), dim = 1)
        #net.to(DEVICE)
        epoch = 1+self.step/len(dataloader)
        tloss = 0
        for i, batch in enumerate(dataloader):
            forwarded=net(batch)
            c = self.cost(forwarded)
            loss = c[0]/lazy_step
            tloss += loss.item()
            loss.backward()
            #print('[Training][Epoch: %d]Step : %d Mean Loss: %.2f Decoding Entropy: %.2f Reconstruct Entropy: %.2f OLoss: %.2f' % (epoch, self.step, c[0], c[1], c[2], c[3]))
            if (i+1)%lazy_step == 0 or i == (len(dataloader) -1):
                #print('update')
                self.optimize()
                print('[Training][Epoch: %d]Step : %d Mean Loss: %.2f' % (epoch, self.step, tloss))
                tloss = 0
            del batch
    
    def run_train(self, data_dir, num_epochs, b_size, check_dir, lazy_step = 1):
        if not os.path.exists(check_dir):
            os.makedirs(check_dir)

        train_dataloader, valid_dataloader, test_dataloader = create_datasets(data_dir, b_size)
        best_val_loss = None
        start = time.time()
        for epoch in range(1,num_epochs+1):
            self.train(train_dataloader, lazy_step)
            with open(check_dir+'/valid.txt', 'a') as f:
                f.write('%s[Epoch:%d]' % (time_since(start, epoch / num_epochs), epoch))
            l = self.validate(valid_dataloader, check_dir+'/valid.txt')
            self.scheduler.step(l)
            #with open(check_dir+'/test.txt', 'a') as f:
            #    f.write('%s[Epoch:%d]' % (time_since(start, epoch / num_epochs), epoch))
            #self.validate(test_dataloader, check_dir+'/test.txt')
            if not best_val_loss or l < best_val_loss:
                with open(check_dir+'/best_epoch', 'wb') as f:
                    torch.save(self, f)
                best_val_loss = l
                #self.optim_e = torch.optim.Adam(self.encode_params, lr=self.lr/2)
                #self.optim_d = torch.optim.Adam(self.decode_params, lr=self.lr/2)
