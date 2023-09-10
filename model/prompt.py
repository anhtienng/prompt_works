import torch
import torch.nn as nn

class EncoderPrompt():
    def __init__(self, type='ctranspath', prompt_len=1, skip_layers=[]) -> None:
        self.prompt_combination = create_prompt_combination(type, prompt_len, skip_layers)

class DecoderPrompt():
    def __init__(self, type='plip', prompt_len=1, skip_layers=[]) -> None:
        self.prompt_combination = create_prompt_combination(type, prompt_len, skip_layers)


def create_prompt_combination(type='ctranspath', prompt_len=1, skip_layers=[]):
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
                prompt_dict[i] = create_prompt_and_key(shape)
            else:
                prompt_dict[i] = None
        key = create_prompt_and_key((1,model_dim))
        return key, prompt_dict
    

def create_prompt_and_key(shape):
    if len(shape) == 3:
        p = torch.nn.Parameter(torch.FloatTensor(shape[0], shape[1], shape[2]), requires_grad=True)
    elif len(shape) == 2:
        p = torch.nn.Parameter(torch.FloatTensor(shape[0], shape[1]), requires_grad=True)
    nn.init.uniform_(p)
    return p