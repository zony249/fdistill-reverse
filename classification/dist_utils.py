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


LAYER_MAP = {
    1: [11], 
    2: [5, 11],
    3: [3, 7, 11], 
    4: [2, 5, 8, 11], 
    6: [1, 3, 5, 7, 9, 11], 
    12: list(range(12)) 
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

# deprecated because this is wrong
class DistPairwise(Measure): 
    def __init__(self): 
        super().__init__() 
        raise ValueError("Deprecated because this is wrong... this assumes the hidden states were already meaned along the sequence lengthm which is incorrect.")
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
    

class MeanPairwiseLayerTransformDist(Measure): 
    def __init__(self):
        super().__init__() 
    def compute(self, hidden_1, hidden_2): 
        """
        hidden_1: torch.Tensor[batch, layers_1, seq_len, hidden]
        hidden_1: torch.Tensor[batch, layers_2, seq_len, hidden]
        produces: torch.Tensor[layers_1, layers_2]
        """
        layers_1 = hidden_1.shape[1]
        layers_2 = hidden_2.shape[1]
        seq_len = min(hidden_1.shape[2], hidden_2.shape[2]) 
        dim = hidden_1.shape[-1]

        per_layer_raw_diff = hidden_1[:, :, None, :seq_len, :] - hidden_2[:, None, :, :seq_len, :] #[batch, layer, layer, seq_len, hidden]
        seq_batch_concat = torch.movedim(per_layer_raw_diff, 3, 1).reshape((-1, layers_1, layers_2, dim)) #[batch x seq_len, layer, layer, hidden]
        norms = seq_batch_concat.norm(dim=-1) #[batch x seq_len, layer, layer]
        means = norms.mean(dim=0)
        self.val = means 
        return self

class MeanPairwiseLayerTransformCosine(Measure): 
    def __init__(self): 
        super().__init__() 
    def compute(self, hidden_1, hidden_2): 
        """
        hidden_1: torch.Tensor[batch, layers_1, seq_len, hidden]
        hidden_1: torch.Tensor[batch, layers_2, seq_len, hidden]
        produces: torch.Tensor[layers_1, layers_2]
        """
        layers_1 = hidden_1.shape[1]
        layers_2 = hidden_2.shape[1]
        seq_len = min(hidden_1.shape[2], hidden_2.shape[2]) 
        dim = hidden_1.shape[-1]

        hid_1_len_trunc = hidden_1[:, :, :seq_len, :]
        hid_2_len_trunc = hidden_2[:, :, :seq_len, :]

        uv = (hid_1_len_trunc[:, :, None, :, :] * hid_2_len_trunc[:, None, :, :, :]).sum(dim=-1) # [batch, layers1, layers2, seq_len]
        uu = (hid_1_len_trunc[:, :, None, :, :]).norm(dim=-1) # [batch, layer1, 1, seq_len]
        vv = (hid_2_len_trunc[:, None, :, :, :]).norm(dim=-1) # [batch, 1, layer2, seq_len]
        cosines_raw = uv / (uu * vv)
        seq_batch_concat = torch.movedim(cosines_raw, 3, 1).reshape(-1, layers_1, layers_2) # [batch x seq, layers_1, layers_2]
        means = seq_batch_concat.mean(dim=0)
        self.val = means 
        return self


class MeanLayerDistance(Measure): 
    def __init__(self): 
        super().__init__()
    def compute(self, pairwise_layer_dist): 
        """
        pairwise_layer_dist: torch.Tensor[layers, layers]
        """
        # if square 
        if pairwise_layer_dist.shape[0] == pairwise_layer_dist.shape[-1]: 
            lower_tril = torch.tril(pairwise_layer_dist, diagonal=-1)
            self.val = lower_tril.sum() / torch.arange(1, pairwise_layer_dist.shape[0]).sum()
        else: 
            self.val = pairwise_layer_dist.mean()
        return self

class MeanPairwiseCosine(Measure): 
    def __init__(self): 
        super().__init__()
    def compute(self, hidden_1, hidden_2): 
        """
        hidden_1: torch.Tensor[batch, layers_1, seq_len, hidden]
        hidden_1: torch.Tensor[batch, layers_2, seq_len, hidden]
        produces: torch.Tensor[layers_1, layers_2]
        """
        layers_1 = hidden_1.shape[1]
        layers_2 = hidden_2.shape[1]
        seq_len = min(hidden_1.shape[2], hidden_2.shape[2]) 
        dim = hidden_1.shape[-1]

        hidden_1 = hidden_1[:, :, :seq_len, :]
        hidden_2 = hidden_2[:, :, :seq_len, :]

        mean1 = hidden_1.mean(dim=(0, 1, 2), keepdim=True)
        mean2 = hidden_2.mean(dim=(0, 1, 2), keepdim=True)
        centered_1 = (hidden_1)[:, :, None, :, :]
        centered_2 = (hidden_2)[:, None, :, :, :]

        uv = (centered_1 * centered_2).sum(dim=-1) #[batch, layer1, layer2, seq_len]
        uu = centered_1.norm(dim=-1)
        vv = centered_2.norm(dim=-1)

        cosines = torch.movedim(uv / (uu * vv), 3, 1).reshape(-1, layers_1, layers_2) #[batch x seq_len, layer1, layer2]
        cosines = cosines.mean(dim=0)
        self.val = cosines 
        return self


class MeanNormedPairwiseCosine(Measure): 
    def __init__(self): 
        super().__init__()
    def compute(self, hidden_1, hidden_2): 
        """
        hidden_1: torch.Tensor[batch, layers_1, seq_len, hidden]
        hidden_1: torch.Tensor[batch, layers_2, seq_len, hidden]
        produces: torch.Tensor[layers_1, layers_2]
        """
        layers_1 = hidden_1.shape[1]
        layers_2 = hidden_2.shape[1]
        seq_len = min(hidden_1.shape[2], hidden_2.shape[2]) 
        dim = hidden_1.shape[-1]

        hidden_1 = hidden_1[:, :, :seq_len, :]
        hidden_2 = hidden_2[:, :, :seq_len, :]

        mean1 = hidden_1.mean(dim=(0, 1, 2), keepdim=True)
        mean2 = hidden_2.mean(dim=(0, 1, 2), keepdim=True)
        mean = (hidden_1.shape[1] * mean1 + hidden_2.shape[1] * mean2) / (hidden_1.shape[1] + hidden_2.shape[1])
        centered_1 = (hidden_1 - mean)[:, :, None, :, :]
        centered_2 = (hidden_2 - mean)[:, None, :, :, :]

        uv = (centered_1 * centered_2).sum(dim=-1) #[batch, layer1, layer2, seq_len]
        uu = centered_1.norm(dim=-1)
        vv = centered_2.norm(dim=-1)

        cosines = torch.movedim(uv / (uu * vv), 3, 1).reshape(-1, layers_1, layers_2) #[batch x seq_len, layer1, layer2]
        cosines = cosines.mean(dim=0)
        self.val = cosines 
        return self

class MeanNormedCosine(Measure): 
    def __init__(self, between_model=False):
        super().__init__() 
        self.layer_map = LAYER_MAP
        self.between_model = between_model
    def compute(self, pairwise_cosines): 
        """
        pairwise_cosines: torch.Tensor[layers_1, layers_2]
        """
        if self.between_model:
            print(pairwise_cosines.shape)
            num_teacher_layers = pairwise_cosines.shape[0]-1
            num_student_layers = pairwise_cosines.shape[1]-1 
            layer_mapping = [0] + [i + 1 for i in self.layer_map[num_student_layers]] if num_student_layers != num_teacher_layers else [0] + [i+1 for i in range(num_student_layers)]
            print(layer_mapping)
            total = 0
            for j in range(num_student_layers): 
                i = layer_mapping[j] 
                total += pairwise_cosines[i, :].mean()
            total /= num_student_layers
            self.val = total 
            return self
        else: 
            #between layer 
            if pairwise_cosines.shape[0] == pairwise_cosines.shape[-1]: 
                lower_tril = torch.tril(pairwise_cosines, diagonal=-1)
                self.val = lower_tril.sum() / torch.arange(1, pairwise_cosines.shape[0]).sum()
            else: 
                self.val = pairwise_cosines.mean()
            return self

class StructuredCosine(Measure): 
    def __init__(self): 
        super().__init__() 
    def compute(self, hidden_t, hidden_s):
        """
        cosine of each corresponding transformation of student and teacher. 
        Deeper teachers will have many of their hidden states ignored

        hidden_t: torch.Tensor[batch, layers_t, seq_len, hidden]
        hidden_s: torch.Tensor[batch, layers_s, seq_len, hidden]
        produces: torch.Tensor[layers_s, layers_s]
        """
        num_stu_layers = hidden_s.shape[1]-1
        teacher_matched_layers = [0] + [i+1 for i in LAYER_MAP[num_stu_layers]] if hidden_t.shape[1] != hidden_s.shape[1] else list(range(hidden_s.shape[1]))

        layers_t = hidden_t.shape[1]
        layers_s = hidden_s.shape[1]
        seq_len = min(hidden_t.shape[2], hidden_s.shape[2]) 
        dim = hidden_t.shape[-1]

        hidden_t_select = hidden_t[:, :, :seq_len][:, teacher_matched_layers]
        hidden_s_trunc = hidden_s[:, :, :seq_len]
        assert hidden_t_select.shape[1] == hidden_s_trunc.shape[1], "number of hidden layers don't match"

        print(hidden_t_select.shape)
        print(hidden_s_trunc.shape)

        hidden_t_grid = hidden_t_select[:, :, None, :, :] - hidden_t_select[:, None, :, :, :]
        hidden_s_grid = hidden_s_trunc[:, :, None, :, :] - hidden_s_trunc[:, None, :, :, :]

        uv = (hidden_t_grid * hidden_s_grid).sum(dim=-1) # [batch, l, l, seq_len]
        uu = hidden_t_grid.norm(dim=-1) 
        vv = hidden_s_grid.norm(dim=-1) 

        cosines = torch.movedim(uv / (uu * vv + 1e-6), -1, 1).reshape(-1, layers_s, layers_s).mean(dim=0)
        self.val = cosines 
        return self

class StructuredCosineHistogram(Measure): 
    def __init__(self): 
        super().__init__() 
    def compute(self, struct_cos: torch.Tensor): 
        """ 
        struct_cos: torch.Tensor[layers, layers] 
        """ 
        mask = torch.tril(torch.ones_like(struct_cos, device=struct_cos.device), diagonal=-1) > 0.5
        elems = torch.masked_select(struct_cos, mask=mask)
        self.val = elems
        # print(self.val)
        return self

class CosineFromStudent(Measure): 
    def __init__(self, from_student_layer): 
        self.from_student_layer = from_student_layer
        super().__init__() 
    def compute(self, hidden_t, hidden_s): 
        """
        cos(T1-S1, Tk-S1)
        hidden_t: torch.Tensor[batch, layers_t, seq_len, hidden]
        hidden_s: torch.Tensor[batch, layers_s, seq_len, hidden]
        """

        layers_t = hidden_t.shape[1]
        layers_s = hidden_s.shape[1]
        seq_len = min(hidden_t.shape[2], hidden_s.shape[2]) 
        dim = hidden_t.shape[-1]

        assert hidden_s.shape[-1] == dim 
        hidden_t_trunc = hidden_t[:, :, :seq_len]
        hidden_s_trunc = hidden_s[:, :, :seq_len]

        cos_sims = []
        ht1 = hidden_t_trunc[:, 1].reshape(-1, dim)
        for i in range(layers_t): 
            htk = hidden_t_trunc[:, i].reshape(-1, dim)
            hs1 = hidden_s_trunc[:, self.from_student_layer].reshape(-1, dim)

            u = ht1 - hs1
            v = htk - hs1
            uv = (u * v).sum(dim=-1)
            uu = u.norm(dim=-1) 
            vv = v.norm(dim=-1)
            cosine = (uv / (uu * vv + 1e-6))
            cos_sims = cosine.tolist()  
            print(cosine)
        self.val = cos_sims 
        return self
    def accum(self): 
        if self.counter == 0: 
            self.mean = self.val 
        else: 
            self.mean += self.val
        self.counter += 1




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
    """
    returns: seq: torch.Tensor[batch_size, layers, seq_len, hidden]
    """
    return torch.stack(raw_encoder_outputs, dim=1)
def tensorize_decoder_outputs(raw_decoder_outputs): 
    return tensorize_encoder_outputs(raw_decoder_outputs)




if __name__ == "__main__": 
    df = px.data.iris()
    print(df)
    fig = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_width',
              color='species')
    fig.show()