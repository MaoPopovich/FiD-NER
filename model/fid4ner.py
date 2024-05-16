import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.crf import CRF
from .bert4fid import Bert4FiD,ScaledDotProductAttention
from transformers import BertPreTrainedModel,BertConfig,BertModel

class FiD4Ner(BertPreTrainedModel):
    def __init__(self, config):
        super(FiD4Ner, self).__init__(config)
        self.bert4fid = Bert4FiD(config)
        self.n_passages = config.n_passages
        self.max_seq_length = config.max_seq_length
        self.fusion = ScaledDotProductAttention(d_model=config.hidden_size,d_k=512,d_v=512)
        # self.fusion = nn.Linear(self.n_passages*self.max_seq_length, self.max_seq_length)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()
    
    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.normal_(m.weight, std=0.001)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,labels=None):
        '''
        bert -> fusion -> classifier -> CRF
        '''
        # print('input_ids.shape:', input_ids.shape)
        outputs =self.bert4fid(input_ids = input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids,return_dict=False)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        
        # Fusion based on attention between sentences 
        
        sequence_main = sequence_output[:,:self.max_seq_length,:]
        sequence_context = sequence_output[:,self.max_seq_length:,:]

        attn_mask = attention_mask[:,0,:]
        attn_output = self.fusion(queries=sequence_main,keys=sequence_context,values=sequence_context,attention_mask=attn_mask)
        fusion_output = torch.add(torch.mul(sequence_main,0.8),torch.mul(attn_output,0.2))
        # fusion_output = torch.mul(sequence_main,0.8)

        # fusion using a simple Linear
        # x = torch.transpose(sequence_output,1,2)
        # y = self.fusion(x)
        # y = self.dropout(y)
        # fusion_output = torch.transpose(y,1,2)
        

        # project hidden_state to tag space
        logits = self.classifier(fusion_output)
        outputs = (logits,)
        if labels is not None:
            loss = self.crf(emissions = logits, tags=labels, mask=attention_mask[:,0,:])
            outputs =(-1*loss,)+outputs
        return outputs # (loss), scores


if __name__ == '__main__':
    batch_size, n_passages, max_len = 4,7,32
    input_ids = torch.zeros((batch_size,n_passages,max_len),dtype=torch.long)
    attention_mask = torch.ones((batch_size,n_passages,max_len),dtype=torch.long)
    token_type_ids = torch.zeros((batch_size,n_passages,max_len),dtype=torch.long)

    config = BertConfig.from_pretrained('bert-base-chinese')
    model = FiD4Ner(config)
    output = model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
    print("result:\t",output[0].shape)