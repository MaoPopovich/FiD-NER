# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import types
import torch
from transformers import BertModel,BertConfig
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
import numpy as np

class Bert4FiD(BertModel):
    def __init__(self, config):
        super().__init__(config)
        self.wrap_encoder() 

    def forward_(self, **kwargs):
        if 'input_ids' in kwargs:
            kwargs['input_ids'] = kwargs['input_ids'].view(kwargs['input_ids'].size(0), -1)
        if 'attention_mask' in kwargs:
            kwargs['attention_mask'] = kwargs['attention_mask'].view(kwargs['attention_mask'].size(0), -1)
        if 'token_type_ids' in kwargs:
            kwargs['token_type_ids'] = kwargs['token_type_ids'].view(kwargs['token_type_ids'].size(0), -1)

        return super(Bert4FiD, self).forward(
            **kwargs
        )

    # We need to resize as B x (N * L) instead of (B * N) x L here
    # because the T5 forward method uses the input tensors to infer
    # dimensions used in the decoder.
    # EncoderWrapper resizes the inputs as (B * N) x L.
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,**kwargs):
        if input_ids != None:
            # inputs might have already be resized in the generate method
            if input_ids.dim() == 3:
                self.encoder.n_passages = input_ids.size(1)
            input_ids = input_ids.view(input_ids.size(0),-1)
        if attention_mask != None:
            attention_mask = attention_mask.view(attention_mask.size(0),-1)
        if token_type_ids != None:
            token_type_ids = token_type_ids.view(token_type_ids.size(0),-1)
        
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs
        )

    # We need to resize the inputs here, as the generate method expect 2D tensors
    # def generate(self, input_ids, attention_mask, max_length):
    #     self.encoder.n_passages = input_ids.size(1)
    #     return super().generate(
    #         input_ids=input_ids.view(input_ids.size(0), -1),
    #         attention_mask=attention_mask.view(attention_mask.size(0), -1),
    #         max_length=max_length
    #     )

    def wrap_encoder(self, use_checkpoint=False):
        """
        Wrap Bert encoder to obtain a Fusion-in-Decoder model.
        """
        self.encoder = EncoderWrapper(self.encoder, use_checkpoint=use_checkpoint)

    def unwrap_encoder(self):
        """
        Unwrap Fusion-in-Decoder encoder, useful to load T5 weights.
        """
        self.encoder = self.encoder.encoder
        layer = []
        for mod in self.encoder.layer:
            layer.append(mod.module)
        layer = nn.ModuleList(layer)
        self.encoder.layer = layer

    def load_bert(self, state_dict):
        self.unwrap_encoder()
        self.load_state_dict(state_dict)
        self.wrap_encoder()

    def set_checkpoint(self, use_checkpoint):
        """
        Enable or disable checkpointing in the encoder.
        See https://pytorch.org/docs/stable/checkpoint.html
        """
        for mod in self.encoder.encoder.layer:
            mod.use_checkpoint = use_checkpoint


class EncoderWrapper(torch.nn.Module):
    """
    Encoder Wrapper for Bert Wrapper to obtain a Fusion-in-Decoder model.
    """
    def __init__(self, encoder, use_checkpoint=False):
        super().__init__()

        self.encoder = encoder
        apply_checkpoint_wrapper(self.encoder, use_checkpoint)

    def forward(self, input_ids=None, attention_mask=None, **kwargs,):
        # total_length = n_passages * passage_length 
        bsz, total_length, _ = input_ids.shape   # bsz, total_length, embd_size = input_ids.shape
        passage_length = total_length // self.n_passages
        input_ids = input_ids.view(bsz*self.n_passages, passage_length,-1)
        attention_mask = attention_mask.view(bsz*self.n_passages, -1, 1, passage_length)
        outputs = self.encoder(input_ids, attention_mask, **kwargs)
        outputs = (outputs[0].view(bsz, self.n_passages*passage_length, -1), ) + outputs[1:]
        return outputs

class CheckpointWrapper(torch.nn.Module):
    """
    Wrapper replacing None outputs by empty tensors, which allows the use of
    checkpointing.
    """
    def __init__(self, module, use_checkpoint=False):
        super().__init__()
        self.module = module
        self.use_checkpoint = use_checkpoint

    def forward(self, hidden_states, attention_mask, layer_head_mask, 
                encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions,**kwargs):
        if self.use_checkpoint and self.training:
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            def custom_forward(*inputs):
                output = self.module(*inputs, **kwargs)
                empty = torch.tensor(
                    [],
                    dtype=torch.float,
                    device=output[0].device,
                    requires_grad=True)
                output = tuple(x if x is not None else empty for x in output)
                return output

            output = torch.utils.checkpoint.checkpoint(
                custom_forward,
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )
            output = tuple(x if x.size() != 0 else None for x in output)
        else:
            output = self.module(hidden_states, attention_mask, layer_head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions, **kwargs)
        return output

def apply_checkpoint_wrapper(bertencoder, use_checkpoint):
    """
    Wrap each block of the encoder to enable checkpointing.
    """
    layer = []
    for mod in bertencoder.layer:
        wrapped_mod = CheckpointWrapper(mod, use_checkpoint)
        layer.append(wrapped_mod)
    layer = nn.ModuleList(layer)
    bertencoder.layer = layer


class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def  __init__(self,d_model,d_k,d_v,dropout=.1):
        '''
        param d_model: Output dimensionality of the model
        param d_k: Dimensionality of queries and keys
        param d_v: Dimensionality of values
        '''
        super(ScaledDotProductAttention,self).__init__()
        self.fc_q = nn.Linear(d_model,d_k)
        self.fc_k = nn.Linear(d_model,d_k)
        self.fc_v = nn.Linear(d_model,d_v)
        self.fc_o = nn.Linear(d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        self.init_weights()
    

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.d_k)  # (b_s, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.d_k).permute(0, 2, 1)  # (b_s, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.d_v)  # (b_s, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            extended_attention_mask = attention_mask[:,:,None]
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            att = att + extended_attention_mask
        att = F.softmax(att, dim=-1)
        att=self.dropout(att)

        out = torch.matmul(att, v)  # (b_s, nq, d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out




if __name__ == '__main__':
    context=torch.randn(50,49,768)
    query = torch.randn(50,56,768)
    attention_mask = torch.ones(50,56)
    sa = ScaledDotProductAttention(d_model=768, d_k=512, d_v=512)
    output=sa(query,context,context,attention_mask)
    print(output.shape)
