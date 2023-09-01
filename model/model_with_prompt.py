import torch
import torch.nn as nn
from .swin_transformer import ctranspath
from .prompt import Prompt

class PromptModel(nn.Module):
    def __init__(self, encoder_type='ctranspath', 
                 prompt_len=1,
                 enconder_ckpt_path='/home/compu/anhnguyen/TransPath/ctranspath.pth',
                 skip_layers=[],
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if encoder_type == 'ctranspath':
            self.encoder = ctranspath()
            self.encoder.head = nn.Identity()
            self.load_encoder(enconder_ckpt_path)
        else:
            raise ValueError(f'{encoder_type} is not supported')
        self.prompt = Prompt(encoder_type, prompt_len, skip_layers)
        self.key, self.prompt_dict = self.prompt.prompt_combination
        for layer_id in self.prompt_dict:
            setattr(self.encoder, f'prompt_layer_{layer_id}', self.prompt_dict[layer_id])
                
    def load_encoder(self, ckpt_path, freeze=True):
        td = torch.load(ckpt_path)
        self.encoder.load_state_dict(td['model'], strict=True)
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def get_query(self, x):
        with torch.no_grad():
            query = self.encoder(x, use_prompt=False)
        return query

    def forward(self, x):
        x = self.encoder(x, use_prompt=True)

        return x
