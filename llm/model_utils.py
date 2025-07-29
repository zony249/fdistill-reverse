from typing import Tuple, Dict, List, Union, Optional
import torch  
from torch import nn
from copy import deepcopy
from transformers import (
    PreTrainedModel, 
    PreTrainedTokenizerBase, 
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoConfig, 
)
from transformers.models.qwen3 import Qwen3Config, Qwen3ForCausalLM

LAYER_COPY_MAP = { 
    28: [0, 4, 9, 13, 18, 22, 27], 
    36: [0, 4, 9, 13, 18, 22, 27, 31, 35], 
    40: [0, 4, 9, 13, 18, 22, 27, 31, 35, 39]
}


def load_model(hf_name_or_path: str,
               **kwargs) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]: 
    config = AutoConfig.from_pretrained(hf_name_or_path, **kwargs) 
    tok = AutoTokenizer.from_pretrained(hf_name_or_path) 
    
    model = AutoModelForCausalLM.from_pretrained(hf_name_or_path, config=config, **kwargs)

    return model, tok


def create_student_from_teacher(hf_name_or_path: str, mode="weight_copy"): 

    config = AutoConfig.from_pretrained(hf_name_or_path)
    config.torch_dtype = torch.bfloat16
    tok = AutoTokenizer.from_pretrained(hf_name_or_path)

    if config.model_type == "qwen3": 
        template_model = AutoModelForCausalLM.from_pretrained(hf_name_or_path, torch_dtype=torch.bfloat16) 
        config.num_hidden_layers = len(LAYER_COPY_MAP[template_model.config.num_hidden_layers])
        if mode == "weight_copy":
            
            layers = nn.ModuleList([deepcopy(template_model.model.layers[i]) for i in LAYER_COPY_MAP[template_model.config.num_hidden_layers]])
            embeddings = deepcopy(template_model.model.embed_tokens)
            norm = deepcopy(template_model.model.norm)
            rotary_emb = deepcopy(template_model.model.rotary_emb)
            lm_head = deepcopy(template_model.lm_head)

            empty_model = Qwen3ForCausalLM(config)
            empty_model.model.embed_tokens = embeddings 
            empty_model.model.layers = layers 
            empty_model.model.rotary_emb = rotary_emb 
            empty_model.model.norm = norm 
            empty_model.lm_head = lm_head 

            return empty_model, tok 

        elif mode == "random_init": 
            template_model = Qwen3ForCausalLM(config)
            return template_model, tok
    else: 
        raise NotImplementedError("layer selection is not implemented for the current model")