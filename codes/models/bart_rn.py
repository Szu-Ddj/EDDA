import torch
import torch.nn as nn
from transformers import BertModel, BartConfig, BartForSequenceClassification
from transformers.models.bart.modeling_bart import BartEncoder, BartPretrainedModel

# BART
class Encoder(BartPretrainedModel):
    
    def __init__(self, config: BartConfig):
        
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)
        self.encoder = BartEncoder(config, self.shared)

    def forward(self, input_ids, attention_mask=None, output_attentions=False, output_hidden_states=False, return_dict=False):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return encoder_outputs
class MultiAttRule(nn.Module):
    def __init__(self,hid_dim,heads=4,batch_first=True):
        super(MultiAttRule, self).__init__()
        self.att = nn.MultiheadAttention(hid_dim,heads,batch_first=batch_first)
        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)
        
    def forward(self,q,k,v):
        return self.att(self.w_q(q),self.w_k(k),self.w_v(v))
class bart_classifier_rn(nn.Module):

    def __init__(self,opt):

        super(bart_classifier_rn, self).__init__()
        num_labels, dropout, pretrained_name,max_len = opt.polarities_dim, opt.dropout,opt.pretrained_name, opt.max_seq_len
        self.dropout = nn.Dropout(dropout) 
        self.relu = nn.GELU()
        
        self.config = BartConfig.from_pretrained(pretrained_name)
        self.bart = Encoder.from_pretrained(pretrained_name)
        self.bart.pooler = None
        self.linear = nn.Linear(self.bart.config.hidden_size*2, self.bart.config.hidden_size)
        self.fc = nn.Linear(self.bart.config.hidden_size * 2, self.bart.config.hidden_size)
        self.att = MultiAttRule(self.bart.config.hidden_size) 
        # self.linear = nn.Linear(max_len, 1)
        # self.fc = nn.Linear(self.bart.config.hidden_size, self.bart.config.hidden_size)
        self.out = nn.Linear(self.bart.config.hidden_size, num_labels)
        
    def forward(self, input):
        
        x_input_ids, x_atten_masks,r_input_ids, r_atten_masks = input
        last_hidden = self.bart(input_ids=x_input_ids, attention_mask=x_atten_masks)[0]
        r_last_hidden = self.bart(input_ids=r_input_ids, attention_mask=r_atten_masks)[0]
        eos_token_ind = x_input_ids.eq(self.config.eos_token_id).nonzero()
        if len(eos_token_ind) != 3*len(x_input_ids):
            out = torch.sum(last_hidden,1) / x_atten_masks.sum(1).to('cuda').unsqueeze(1)
            out = self.out(out)
            return out

        b_eos = [eos_token_ind[i][1] for i in range(len(eos_token_ind)) if i%3==0]
        e_eos = [eos_token_ind[i][1] for i in range(len(eos_token_ind)) if (i+1)%3==0]
        x_atten_clone = x_atten_masks.clone().detach()
        for begin, end, att, att2 in zip(b_eos, e_eos, x_atten_masks, x_atten_clone):
            att[begin:], att2[:begin+2] = 0, 0 # att all </s> --> 0; att2 1st and 2nd </s> --> 0
            att[0], att2[end] = 0, 0 # <s> --> 0; 3rd </s> --> 0
        
        txt_l = x_atten_masks.sum(1).to('cuda')
        topic_l = x_atten_clone.sum(1).to('cuda')
        # rx_l = r_atten_masks.sum(1).to('cuda')
        txt_l += 1
        txt_vec = x_atten_masks.type(torch.FloatTensor).to('cuda')
        topic_vec = x_atten_clone.type(torch.FloatTensor).to('cuda')
        # rx_vec = r_atten_masks.type(torch.FloatTensor).to('cuda')
        txt_mean = torch.einsum('blh,bl->bh', last_hidden, txt_vec) / txt_l.unsqueeze(1)
        topic_mean = torch.einsum('blh,bl->bh', last_hidden, topic_vec) / topic_l.unsqueeze(1)
        # rx_mean = torch.einsum('blh,bl->bh', r_last_hidden, rx_vec) / rx_l.unsqueeze(1)
        cat = torch.cat((txt_mean, topic_mean), dim=1)
        rout,_score = self.att(self.fc(cat).unsqueeze(1),r_last_hidden,r_last_hidden)
        rout = nn.ReLU()(rout.squeeze(1))
        query = self.dropout(cat)
        linear = self.relu(self.linear(query))
        out = self.out(0.1 * rout + linear)
        
        return out