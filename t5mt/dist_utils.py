import os 
from typing import List, Dict, Tuple, Union, Any, Optional 

import numpy as np 
import torch  
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd

COLOURS = {
    0: "black", 
    1: "brown",
    2: "red", 
    3: "orange",
    4: "yellow", 
    5: "green", 
    6: "blue", 
    7: "violet", 
    8: "gray", 
    9: "turquoise", 
    10: "mediumslateblue", 
    11: "purple", 
    12: "darkslategray"
}


class Measure: 
    def __init__(self): 
        self.val=0
        self.mean = 0
        self.var = 0
        self.counter = 0
    def accum(self): 
        self.counter += 1 
        self.mean = ((self.counter - 1) / self.counter) * self.mean + (1 / self.counter) * self.val
        self.var = ((self.counter - 1) / self.counter) * self.var + (1/self.counter) * (self.val - self.mean) * (self.val - self.mean)
        return self
    def __call__(self, *args, **kwargs): 
        return self.compute(*args, **kwargs)
    def compute(self): 
        raise NotImplementedError()
    def get_val(self):
        return self.val
    def get_mean(self):
        return self.mean
class DistBetweenModels(Measure): 
    def __init__(self):
        self.val = 0 
        super().__init__() 
    def compute(self, hidden_t, hidden_s): 
        """
        hidden_t: torch.Tensor[batch, layers, hidden]
        hidden_s: torch.Tensor[batch, layers, hidden]
        produces: torch.Tensor[layers]: distance between teacher and student layers.
        """
        self.val = (hidden_t - hidden_s).norm(dim=-1).mean(dim=0)
        return self
class DistBetweenLayers(Measure): 
    def __init__(self): 
        super().__init__() 
    def compute(self, hidden_states): 
        """
        hidden_states: torch.Tensor[batch, layer, hidden]
        produces: torch.Tensor[layers]: distance between layers.
        """
        top = hidden_states[:, 1:, :]
        bottom = hidden_states[:, :-1, :]
        self.val = (top - bottom).norm(dim=-1).mean(dim=0)
        return self
class DistPairwise(Measure): 
    def __init__(self): 
        super().__init__() 
    def compute(self, hidden_1, hidden_2):
        """
        hidden_X: torch.Tensor[batch, layers, hidden]
        """
        a = hidden_1[:, :, None, :] - hidden_2[:, None, :, :]
        self.val = a.norm(dim=-1).mean(dim=0) 
        return self

class PerLayerSTDDEV(Measure): 
    def __init__(self): 
        super().__init__()
    def compute(self, layer_outputs_1, layer_outputs_2): 
        """
        layer_outputs: torch.Tensor[batch, layers, seq_len, hidden]
        """
        self.val = (layer_outputs_1 - layer_outputs_2).std(dim=-1).mean(dim=(0, 2))
        return self

class TransformDistance(Measure): 
    def __init__(self):
        self.val = 0 
        super().__init__() 
    def compute(self, hidden_1, hidden_2): 
        """
        hidden_t: torch.Tensor[batch, layers, seq_len, hidden]
        hidden_s: torch.Tensor[batch, layers, seq_len, hidden]
        produces: torch.Tensor[layers]: distance between teacher and student layers.
        """
        self.val = (hidden_1 - hidden_2).norm(dim=-1).mean(dim=(0, 2))
        return self





def compute_pca(states:List[torch.Tensor]) -> torch.Tensor: 
    """
    states: List[torch.Tensor[num_samples, hidden dimension]]
    """

    x = torch.cat(states, dim=0)
    x_u = x - x.mean(dim=0)
    cov = torch.matmul(x_u.T, x_u) / x_u.shape[0] 
    U, S, V = torch.svd(cov) 
    U_red = U[:, :3] 
    print("variance explained: ", (S[:3].sum()/ S.sum()).item())
    return U_red 


def pca_transform(inputs, PCA_mat):
    """
    inputs: torch.Tensor[*, input_dim]
    PCA_mat: torch.Tensor[input_dim, output_dim]
    """    


    return torch.matmul(inputs, PCA_mat)


def create_display_df(enc_pca: torch.Tensor, dec_pca: torch.Tensor, savedir: str):
    """
    inputs: torch.Tensor[num_samples, enc/dec, layers, 3]
    """

    assert not (enc_pca is None and dec_pca is None), "encoder and decoder pca states cannot simulaneously be None"


    df = pd.DataFrame()
    if enc_pca is not None:
        for i in range(enc_pca.shape[1]): 
            enc_layer_samples = torch.cat([enc_pca[:, i, :], i * torch.ones((enc_pca.shape[0], 1))], dim=-1) if enc_pca is not None else None
            layer_samples = torch.cat([enc_layer_samples, torch.zeros((enc_layer_samples.shape[0], 1))], dim=-1)

            if df.empty: 
                df = pd.DataFrame(layer_samples, columns=["x", "y", "z", "layer", "is_decoder"])
            else: 
                df = pd.concat([df, pd.DataFrame(layer_samples, columns=["x", "y", "z", "layer", "is_decoder"])]) 

    if dec_pca is not None: 
        for i in range(dec_pca.shape[1]): 
            dec_layer_samples = torch.cat([dec_pca[:, i, :], i * torch.ones((dec_pca.shape[0], 1))], dim=-1) if dec_pca is not None else None
            layer_samples = torch.cat([dec_layer_samples, torch.ones((dec_layer_samples.shape[0], 1))], dim=-1)

            if df.empty: 
                df = pd.DataFrame(layer_samples, columns=["x", "y", "z", "layer", "is_decoder"])
            else: 
                df = pd.concat([df, pd.DataFrame(layer_samples, columns=["x", "y", "z", "layer", "is_decoder"])]) 

    return df    


def tensorize_decoder_generate_outputs(raw_decoder_outputs, num_beams):
    """
    returns: seq: torch.Tensor[batch_size, layers, seq_len, hidden]
    """
    seq = [] 
    for layers in raw_decoder_outputs:
        l = []
        for states in layers: 
            #states are [batch_size x num_beams, 1, dim]
            states_normed = torch.cat([states[i*num_beams: (i+1)*num_beams].mean(dim=0, keepdims=True) for i in range(int(np.ceil(states.shape[0] / num_beams)))], dim = 0)
            l.append(states_normed)
        l = torch.cat(l, dim=1)
        seq.append(l)
    seq = torch.stack(seq, dim=2)
    return seq


def tensorize_encoder_outputs(raw_encoder_outputs):
    return torch.stack(raw_encoder_outputs, dim=1)
def tensorize_decoder_outputs(raw_decoder_outputs): 
    return tensorize_encoder_outputs(raw_decoder_outputs)



def forward(
    input_ids=None,
    attention_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    inputs_embeds=None,
    head_mask=None,
    encoder_head_mask=None,
    past_key_values=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    model_config=None, 
    is_decoder=False, 
):
    # Model parallel
    use_cache = use_cache if use_cache is not None else model_config.use_cache
    output_attentions = output_attentions if output_attentions is not None else model_config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else model_config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else model_config.use_return_dict

    if input_ids is not None and inputs_embeds is not None:
        err_msg_prefix = "decoder_" if is_decoder else ""
        raise ValueError(
            f"You cannot specify both {err_msg_prefix}inputs and {err_msg_prefix}inputs_embeds at the same time"
        )
    elif input_ids is not None:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
    else:
        err_msg_prefix = "decoder_" if is_decoder else ""
        raise ValueError(f"You have to specify either {err_msg_prefix}inputs or {err_msg_prefix}inputs_embeds")

    if inputs_embeds is None:
        raise ValueError("input_embeds cannot be None")


    batch_size, seq_length = input_shape

    # required mask seq length can be calculated via length of past
    mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

    if use_cache is True:
        assert is_decoder


    if attention_mask is None:
        attention_mask = torch.ones(batch_size, mask_seq_length).to(inputs_embeds.device)
    if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
        encoder_seq_length = encoder_hidden_states.shape[1]
        encoder_attention_mask = torch.ones(
            batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
        )

    # initialize past_key_values with `None` if past does not exist
    if past_key_values is None:
        past_key_values = [None] * len(self.block)

    # ourselves in which case we just need to make it broadcastable to all heads.
    extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, inputs_embeds.device)

    if self.is_decoder and encoder_attention_mask is not None:
        encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
    else:
        encoder_extended_attention_mask = None

    # Prepare head mask if needed
    head_mask = self.get_head_mask(head_mask, self.config.num_layers)
    encoder_head_mask = self.get_head_mask(encoder_head_mask, self.config.num_layers)
    present_key_value_states = () if use_cache else None
    all_hidden_states = () if output_hidden_states else None
    all_attentions = () if output_attentions else None
    all_cross_attentions = () if (output_attentions and self.is_decoder) else None
    position_bias = None
    encoder_decoder_position_bias = None

    hidden_states = self.dropout(inputs_embeds)

    for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
        layer_head_mask = head_mask[i]
        encoder_layer_head_mask = encoder_head_mask[i]
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(hidden_states.device)
            # Ensure that attention_mask is always on the same device as hidden_states
            if attention_mask is not None:
                attention_mask = attention_mask.to(hidden_states.device)
            if position_bias is not None:
                position_bias = position_bias.to(hidden_states.device)
            if encoder_hidden_states is not None:
                encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
            if encoder_extended_attention_mask is not None:
                encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
            if encoder_decoder_position_bias is not None:
                encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
            if layer_head_mask is not None:
                layer_head_mask = layer_head_mask.to(hidden_states.device)
            if encoder_layer_head_mask is not None:
                encoder_layer_head_mask = encoder_layer_head_mask.to(hidden_states.device)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        layer_outputs = layer_module(
            hidden_states,
            attention_mask=extended_attention_mask,
            position_bias=position_bias,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            encoder_decoder_position_bias=encoder_decoder_position_bias,
            layer_head_mask=layer_head_mask,
            encoder_layer_head_mask=encoder_layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        # layer_outputs is a tuple with:
        # hidden-states, key-value-states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
        hidden_states, present_key_value_state = layer_outputs[:2]

        # We share the position biases between the layers - the first layer store them
        # layer_outputs = hidden-states, key-value-states (self-attention weights),
        # (self-attention position bias), (cross-attention weights), (cross-attention position bias)
        position_bias = layer_outputs[2]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
        # append next layer key value states
        if use_cache:
            present_key_value_states = present_key_value_states + (present_key_value_state,)

        if output_attentions:
            all_attentions = all_attentions + (layer_outputs[3],)
            if self.is_decoder:
                all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

        # Model Parallel: If it's the last layer for that device, put things on the next device
        if self.model_parallel:
            for k, v in self.device_map.items():
                if i == v[-1] and "cuda:" + str(k) != self.last_device:
                    hidden_states = hidden_states.to("cuda:" + str(k + 1))
    pre_lm_hidden = hidden_states
    hidden_states = self.final_layer_norm(hidden_states)
    hidden_states = self.dropout(hidden_states)

    # Add last layer
    if output_hidden_states:
        # all_hidden_states = all_hidden_states + (hidden_states,)
        all_hidden_states = all_hidden_states + (pre_lm_hidden,)
        

    if not return_dict:
        return tuple(
            v
            for v in [
                hidden_states,
                present_key_value_states,
                all_hidden_states,
                all_attentions,
                all_cross_attentions,
            ]
            if v is not None
        )
    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        past_key_values=present_key_value_states,
        hidden_states=all_hidden_states,
        attentions=all_attentions,
        cross_attentions=all_cross_attentions,
    )



if __name__ == "__main__": 
    df = px.data.iris()
    print(df)
    fig = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_width',
              color='species')
    fig.show()