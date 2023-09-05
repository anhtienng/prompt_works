import torch
import torch.nn as nn
from transformers import AutoProcessor
from .swin_transformer import ctranspath
from .clip import CLIPModel
from .prompt import EncoderPrompt, DecoderPrompt
from model.projector import MLP

class PromptModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.device

        # Init visual encoder
        if args.encoder_type == 'ctranspath':
            self.encoder = ctranspath()
            self.encoder.head = nn.Identity()
            td = torch.load(args.encoder_ckpt_path)
            self.encoder.load_state_dict(td['model'], strict=True)
        else:
            raise ValueError(f'Encoder {args.encoder_type} is not supported')
        
        # Init projector
        self.projector = MLP(args)

        # Init text decoder
        if args.decoder_type == 'plip':
            self.decoder = CLIPModel.from_pretrained(args.decoder_ckpt_path).text_model
            self.decoder_head = nn.Linear(512, 49408)
        else:
            raise ValueError(f'Decoder {args.decoder_type} is not supported')

        # Freeze encoder and decoder
        self.freeze_encoder_and_decoder()

        # Init prompt for encoder
        self.encoder_prompt = EncoderPrompt(args.encoder_type, args.encoder_prompt_len, args.encoder_skip_layers)
        self.key, self.encoder_prompt_dict = self.encoder_prompt.prompt_combination
        for layer_id in self.encoder_prompt_dict:
            setattr(self.encoder, f'prompt_layer_{layer_id}', self.encoder_prompt_dict[layer_id])
        
        # Init prompt for decoder
        self.decoder_prompt = DecoderPrompt(args.decoder_type, args.decoder_prompt_len, args.decoder_skip_layers)
        self.decoder_prompt_dict = self.decoder_prompt.prompt_combination
        for layer_id in self.decoder_prompt_dict:
            setattr(self.decoder.encoder, f'prompt_layer_{layer_id}', self.decoder_prompt_dict[layer_id])

        # Init tokenizer
        self.tokenizer = AutoProcessor.from_pretrained(args.tokenizer_type)

    def freeze_encoder_and_decoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False

    def get_query(self, x):
        with torch.no_grad():
            query = self.encoder(x, use_prompt=False)
        return query

    def forward(self, img, text):
        assert len(img.shape)==4 and img.shape[1:] == (3,224,224), \
                    f'Expect img input of shape (bs,3,224,3,224) but got {img.shape}'
        img = self.encoder(img, use_prompt=True)
        img = self.projector(img)
        img = img.reshape(img.shape[0], -1, 512)

        assert len(text) == img.shape[0], \
                    f'Expect two inputs have same length, but got img of {img.shape[0]} and text of {len(text)}'
        text = self.tokenizer(text, return_tensors="pt", padding=True)
        input_ids = text['input_ids'].to(self.device)
        attention_mask = text['attention_mask'].to(self.device)
        output = self.decoder(proj_encoder_feature=img, input_ids=input_ids, attention_mask=attention_mask)
        logits = self.decoder_head(output.last_hidden_state)

        return {
            'input_ids': input_ids,
            'last_layer_logits': logits
        }
    
    def forward_decoder(self, proj_encoder_feature, input_ids, attention_mask):
        output = self.decoder(proj_encoder_feature=proj_encoder_feature,
                              input_ids=input_ids, 
                              attention_mask=attention_mask)
        return output
