import torch
import math
import torch.nn as nn

class EncoderPrompt():
    def __init__(self, type='ctranspath', prompt_len=1, skip_layers=[], distinct=False) -> None:
        self.prompt_combination = create_prompt_combination(type, prompt_len, skip_layers, distinct)

class DecoderPrompt():
    def __init__(self, type='plip', prompt_len=1, skip_layers=[], distinct=False) -> None:
        self.prompt_combination = create_prompt_combination(type, prompt_len, skip_layers, distinct)

class Lora():
    def __init__(self, args, module='encoder') -> None:
        self.lora_combination = create_lora_combination(args, module)
    
def create_prompt_combination(type='ctranspath', prompt_len=1, skip_layers=[], distinct=False):
    prompt_dict = {}
    if type == 'ctranspath':
        head_dim = 32
        num_blocks_in_each_stage = [2,2,6,2]
        shape_of_each_stage = [
            (prompt_len, 3*head_dim),  # 96
            (prompt_len, 6*head_dim),  # 192
            (prompt_len, 12*head_dim), # 384
            (prompt_len, 24*head_dim)  # 768
        ]
        i = 0
        for stage, num_blocks in enumerate(num_blocks_in_each_stage):
            for _ in range(num_blocks):
                if i not in skip_layers:
                    if distinct:
                        pk = create_prompt_and_key(shape_of_each_stage[stage])
                        pv = create_prompt_and_key(shape_of_each_stage[stage])
                        prompt_dict[i] = nn.ParameterList([pk, pv])
                    else:
                        prompt_dict[i] = create_prompt_and_key(shape_of_each_stage[stage])
                else:
                    prompt_dict[i] = None
                i+=1

        model_dim = 768
        key = create_prompt_and_key((1,model_dim))
        return key, prompt_dict
    
    elif type == 'e_plip':
        num_layers = 12
        model_dim = 768
        shape = (prompt_len, model_dim)
        for i in range(num_layers):
            if i not in skip_layers:
                if distinct:
                    pk = create_prompt_and_key(shape)
                    pv = create_prompt_and_key(shape)
                    prompt_dict[i] = nn.ParameterList([pk, pv])
                else:
                    prompt_dict[i] = create_prompt_and_key(shape)
            else:
                prompt_dict[i] = None
        key = create_prompt_and_key((1,model_dim))
        return key, prompt_dict

    elif type == 'd_plip':
        num_layers = 12
        model_dim = 512
        shape = (prompt_len, model_dim)
        for i in range(num_layers):
            if i not in skip_layers:
                if distinct:
                    pk = create_prompt_and_key(shape)
                    pv = create_prompt_and_key(shape)
                    prompt_dict[i] = nn.ParameterList([pk, pv])
                else:
                    prompt_dict[i] = create_prompt_and_key(shape)
            else:
                prompt_dict[i] = None
        key = create_prompt_and_key((1,model_dim))
        return key, prompt_dict
    
    elif type == 'gpt2':
        num_layers = 12
        model_dim = 768
        shape = (prompt_len, model_dim)
        for i in range(num_layers):
            if i not in skip_layers:
                if distinct:
                    pk = create_prompt_and_key(shape)
                    pv = create_prompt_and_key(shape)
                    prompt_dict[i] = nn.ParameterList([pk, pv])
                else:
                    prompt_dict[i] = create_prompt_and_key(shape)
            else:
                prompt_dict[i] = None
        key = create_prompt_and_key((1,model_dim))
        return key, prompt_dict

    else:
        raise ValueError(f'Not support this type: {type}')

def create_lora_combination(args, module):
    lora_dict = {}
    if args.encoder_type == 'ctranspath' and module=='encoder':
        head_dim = 32
        num_blocks_in_each_stage = [2,2,6,2]
        dim_of_each_stage = [
            3*head_dim,  # 96
            6*head_dim,  # 192
            12*head_dim, # 384
            24*head_dim  # 768
        ]
        i = 0
        lora_r = args.lora_r
        for stage, num_blocks in enumerate(num_blocks_in_each_stage):
            for _ in range(num_blocks):
                if i not in args.encoder_lora_skip_layers:
                    shape_a = (dim_of_each_stage[stage]*3, lora_r)
                    shape_b = (lora_r, dim_of_each_stage[stage]*3)
                    lora_a_q, lora_b_q = create_lora(shape_a, shape_b)
                    lora_a_k, lora_b_k = create_lora(shape_a, shape_b)
                    lora_a_v, lora_b_v = create_lora(shape_a, shape_b)
                    lora_dict[i] = nn.ParameterList([lora_a_q, lora_b_q])
                else:
                    lora_dict[i] = None
                i+=1

        model_dim = 768
        key = create_prompt_and_key((1,model_dim))
    elif args.encoder_type == 'e_plip' and module=='encoder':
        raise NotImplementedError
    elif args.decoder_type == 'd_plip' and module=='decoder':
        num_layers = 12
        model_dim = 512
        for i in range(num_layers):
            if i not in args.decoder_lora_skip_layers:
                shape_a = (model_dim, args.lora_r)
                shape_b = (args.lora_r, model_dim)
                lora_a_q, lora_b_q = create_lora(shape_a, shape_b)
                lora_a_k, lora_b_k = create_lora(shape_a, shape_b)
                lora_a_v, lora_b_v = create_lora(shape_a, shape_b)
                lora_dict[i] = nn.ParameterList([lora_a_q, lora_b_q,
                                                lora_a_k, lora_b_k,
                                                lora_a_v, lora_b_v])
            else:
                lora_dict[i] = None
        key = create_prompt_and_key((1,model_dim))
    else:
        raise ValueError(f'Not support this type: {type}')
    
    return key, lora_dict

def create_lora(shape_a, shape_b):
    lora_A = torch.nn.Parameter(torch.FloatTensor(shape_a[0], shape_a[1]), requires_grad=True)
    lora_B = torch.nn.Parameter(torch.FloatTensor(shape_b[0], shape_b[1]), requires_grad=True)

    nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))
    nn.init.zeros_(lora_B)

    return lora_A, lora_B

def create_prompt_and_key(shape):
    if len(shape) == 3:
        p = torch.nn.Parameter(torch.FloatTensor(shape[0], shape[1], shape[2]), requires_grad=True)
    elif len(shape) == 2:
        p = torch.nn.Parameter(torch.FloatTensor(shape[0], shape[1]), requires_grad=True)
    nn.init.uniform_(p)
    return p