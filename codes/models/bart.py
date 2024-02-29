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

class bart_classifier(nn.Module):

    def __init__(self,opt):

        super(bart_classifier, self).__init__()
        num_labels, dropout, pretrained_name,max_len = opt.polarities_dim, opt.dropout,opt.pretrained_name, opt.max_seq_len
        self.dropout = nn.Dropout(dropout) 
        self.relu = nn.GELU()
        
        self.config = BartConfig.from_pretrained(pretrained_name)
        self.bart = Encoder.from_pretrained(pretrained_name)
        self.bart.pooler = None
        self.linear = nn.Linear(self.bart.config.hidden_size*2, self.bart.config.hidden_size)
        # self.linear = nn.Linear(max_len, 1)
        # self.fc = nn.Linear(self.bart.config.hidden_size, self.bart.config.hidden_size)
        self.out = nn.Linear(self.bart.config.hidden_size, num_labels)
        
    def forward(self, input):
        
        x_input_ids, x_atten_masks = input
        last_hidden = self.bart(input_ids=x_input_ids, attention_mask=x_atten_masks)[0]
        # print(last_hidden.size())
        # out = self.linear(last_hidden.transpose(1,2)).squeeze(2)
        # # out = nn.GELU()(self.fc(out))
        # # # out = last_hidden[0]
        

        eos_token_ind = x_input_ids.eq(self.config.eos_token_id).nonzero() # tensor([[0,4],[0,5],[0,11],[1,2],[1,3],[1,6]...])
        if len(eos_token_ind) != 3*len(x_input_ids):
            out = torch.sum(last_hidden,1) / x_atten_masks.sum(1).to('cuda').unsqueeze(1)
            out = self.out(out)
            return out

        # assert len(eos_token_ind) == 3*len(x_input_ids)
        b_eos = [eos_token_ind[i][1] for i in range(len(eos_token_ind)) if i%3==0]
        e_eos = [eos_token_ind[i][1] for i in range(len(eos_token_ind)) if (i+1)%3==0]
        x_atten_clone = x_atten_masks.clone().detach()
        for begin, end, att, att2 in zip(b_eos, e_eos, x_atten_masks, x_atten_clone):
            att[begin:], att2[:begin+2] = 0, 0 # att all </s> --> 0; att2 1st and 2nd </s> --> 0
            att[0], att2[end] = 0, 0 # <s> --> 0; 3rd </s> --> 0
        
        txt_l = x_atten_masks.sum(1).to('cuda')
        topic_l = x_atten_clone.sum(1).to('cuda')
        txt_l += 1
        # print(txt_l,topic_l)
        # assert False
        # print(x_atten_masks[0])
        txt_vec = x_atten_masks.type(torch.FloatTensor).to('cuda')
        topic_vec = x_atten_clone.type(torch.FloatTensor).to('cuda')
        # print('last_hidden, txt_vec', last_hidden.size(), txt_vec.size())
        txt_mean = torch.einsum('blh,bl->bh', last_hidden, txt_vec) / txt_l.unsqueeze(1)
        topic_mean = torch.einsum('blh,bl->bh', last_hidden, topic_vec) / topic_l.unsqueeze(1)
        
        cat = torch.cat((txt_mean, topic_mean), dim=1)
        query = self.dropout(cat)
        linear = self.relu(self.linear(query))
        out = self.out(linear)
        
        return out