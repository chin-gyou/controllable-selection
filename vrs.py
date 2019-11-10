from basepara import *

class seqattn(base):
    def __init__(self, em, h_size, d_size, w_size, lr, bi=False):
        super().__init__()
        #word embedding
        self.embedding = nn.Embedding.from_pretrained(em, freeze = False)
        print(self.embedding.weight.requires_grad)
        dc_size = 2*h_size if bi else h_size
        self.attnD = nn.Linear(d_size, dc_size)

        self.encoder = EncoderRNN(em.size(1), h_size, 0.3, bi)
        self.decoder = DecoderRNN(self.embedding, d_size, context_size=dc_size, drop_out = 0.3)
        #optimizer
        self.lr = lr
        self.predpri1 = nn.Linear(dc_size, w_size)
        self.predpri2 = nn.Linear(w_size, 1)
        #self.optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1.2e-6)
        #self.optim = torch.optim.Adam(self.parameters(), lr=lr)
        #self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, 'min', factor = 0.1, patience = 0)
        self.outproj = nn.Linear(dc_size + d_size, em.size(1))
        self.em_out = nn.Linear(em.size(1), em.size(0))
        self.em_out.weight = self.embedding.weight
        #self.outproj = nn.Linear(dc_size + d_size, em.size(1))

    #return the encoded vector of the context
    def encode(self, enc_text, dec_text):
        mask = (enc_text!=PAD).float()
        slen = mask.sum(0)
        text_embed = self.embedding(enc_text)
        enc_states, ht = self.encoder.run(text_embed, slen, self.mode)
        dlen = (dec_text!=PAD).float().sum(0)
        text_states, _ = self.encoder2.run(self.embedding(dec_text),dlen, self.mode)
        t_state = text_states[-1].expand(enc_text.size(0), -1, -1)
        post_in = torch.cat((enc_states, t_state), 2)
        pre_in = enc_states
        predpost = F.sigmoid(self.predpost2(F.relu((self.predpost1(post_in))))).squeeze(-1)
        predpri = F.sigmoid(self.predpri2(F.relu((self.predpri1(pre_in))))).squeeze(-1)
        
        return text_embed, enc_states, ht, predpri, predpost#seq_len*[batch_size*h_size]
        
    def attnvec(self, encs, dec, kmask, umask):
        attnencs = encs
        attndec = self.attnD(dec).unsqueeze(0)
        
        dot_products = torch.sum(attnencs*attndec, -1)#seq_len*batch_size
        #weights = F.softmax(dot_products,0)
        #weights = weights*kmask*(1-umask)
        #weights = softmax_mask(dot_products, kmask*(1-umask))
        weights = softmax_mask(dot_products, kmask*(1-umask))
        c_vec = torch.sum(encs*(weights.unsqueeze(-1)),0)#batch_size*h_size
        _, m_foc = torch.max(weights, 0)

        return m_foc, c_vec, weights
    """
    return the cost of one batch
    dec_text: dec_len*batch_size
    enc_text, enc_fd, enc_pos, enc_rpos: enc_len*batch_size
    """
    def forward(self, batch):
        dec_text = batch[0]
        enc_text = batch[1]
        _, context, ht, predpri, predpost = self.encode(enc_text, dec_text)
        umask = (batch[1]==PAD).float()#seq_len*batch_size
        t_kl = torch.sum(self.kldiv(predpost, predpri)*(1-umask), 0)
        o_loss = 0
        t_loss = 0
        t_len=torch.sum((dec_text!=PAD).float())
        pad_mask = (dec_text!=PAD).float()
        e_len = torch.sum(1-umask)
        bisample = torch.bernoulli(predpost)
        selected = torch.sum(context*(bisample.unsqueeze(-1)), 0)/(1e-10+bisample.sum(0, True).t())
        d_hidden = nonlinear(self.decoder.initS(selected))
        c_hidden = nonlinear(self.decoder.initC(selected))
        #likelihood for sampled select
        for i in range(dec_text.size(0)):
            _, c_vec, _ = self.attnvec(context, d_hidden, bisample, umask)
            output = self.em_out(self.outproj(torch.cat((d_hidden, c_vec), 1)))
            t_prob = F.softmax(output, -1)
            tg_prob = torch.gather(t_prob, 1, dec_text[i].unsqueeze(-1)).squeeze()
            del t_prob
            o_loss -= torch.log(1e-10 + tg_prob)*pad_mask[i]
            
            d_hidden, c_hidden = self.decoder(dec_text[i], d_hidden, c_hidden, c_vec, self.mode)
        selected = torch.sum(context*(predpost.unsqueeze(-1)), 0)/(1e-10+predpost.sum(0, True).t())
        d_hidden = nonlinear(self.decoder.initS(selected))
        c_hidden = nonlinear(self.decoder.initC(selected))
        #likelihood for soft baseline
        for i in range(dec_text.size(0)):
            _, c_vec, _ = self.attnvec(context, d_hidden, predpost, umask)
            output = self.em_out(self.outproj(torch.cat((d_hidden, c_vec), 1)))
            t_prob = F.softmax(output, -1)
            tg_prob = torch.gather(t_prob, 1, dec_text[i].unsqueeze(-1)).squeeze()
            del t_prob
            t_loss -= torch.log(1e-10 + tg_prob)*pad_mask[i]
            
            d_hidden, c_hidden = self.decoder(dec_text[i], d_hidden, c_hidden, c_vec, self.mode)
        s_r = -o_loss
        b_r = -t_loss
        s_prob = predpri*bisample + (1-predpri)*(1-bisample)
        s_prob = torch.sum(torch.log(s_prob)*(1-umask), 0)
        
        return -(s_prob*(s_r.detach()-b_r.detach())).sum()/t_len + o_loss.sum()/t_len + torch.max(t_kl/t_len, torch.tensor([KL_THRESH*e_len/t_len]).to(DEVICE)), torch.sum(o_loss)/t_len, t_kl/t_len, torch.sum(bisample*(1-umask))/e_len
        #sys.exit()
    
    def cost(self, forwarded):
        return forwarded

    def decode(self, batch, decode_length = 50):
        context = self.encode(batch)
        decoder_outputs = []
        umask = (batch[1]==PAD).float()#seq_len*batch_size
        
        d_hidden = nonlinear(self.decoder.initS(context[-1]))
     
        for i in range(decode_length):
            c_vec = self.attnvec(context, d_hidden, umask, batch)
            o = self.decoder.decode(d_hidden, c_vec)
            topv, topi = o.topk(1)
            topi.squeeze_(-1)
            decoder_outputs.append(topi)
            dec_input = topi

            d_hidden = self.decoder(dec_input, d_hidden, c_vec, 2)
        return torch.stack(decoder_outputs).t()#batch_size*decode_length
        
if __name__ == '__main__':
    em = torch.from_numpy(pickle.load(open(embed, 'rb')))
    s=seqattn(em, h_size, d_size, w_size, lr, True)
    #s=torch.load(open('./seq2seq/xsum/predbillion/best_epoch','rb'))
    print(s.lr)
    print(s.encoder.bi)
    s=s.to(DEVICE)
    parameters = filter(lambda p: p.requires_grad, s.parameters())
    s.optim = torch.optim.Adam(parameters, lr=lr, weight_decay=1.2e-6)
    s.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(s.optim, 'min', factor = 0.1, patience = 1, min_lr = 1e-6)
    #dloader = DataLoader(tabledata(data_dir, 'test'), batch_size = b_size, shuffle=False, collate_fn = merge_sample)
    #s.output_decode(dloader, './attngreedy', data_dir + '/bpe30k.model')

    #s.validate(dloader, './test.txt')
    s.run_train(data_dir, num_epochs=30, b_size = b_size, check_dir = check_dir, lazy_step = lazy_step)
