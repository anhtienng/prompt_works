import torch
import torch.nn as nn
from transformers import AutoProcessor, GPT2Tokenizer
from .swin_transformer import ctranspath
from .clip import CLIPModel
from .gpt2 import GPT2Model
from .prompt import EncoderPrompt, DecoderPrompt, Lora
from model.projector import MLP, MLP_for_prompt

class PromptModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.device

        # Init model components
        self._init_encoder()
        self._init_projector()
        self._init_decoder()
        if args.type not in ['full_ft','single_encoder']:
            self._freeze_encoder_and_decoder()
            self._init_prompt()
        self._init_tokenizer()

    def _init_encoder(self):
        if self.args.encoder_type == 'ctranspath':
            self.encoder = ctranspath()
            self.encoder.head = nn.Identity()
            td = torch.load(self.args.encoder_ckpt_path)
            self.encoder.load_state_dict(td['model'], strict=True)
        elif self.args.encoder_type == 'e_plip':
            self.encoder = CLIPModel.from_pretrained(self.args.decoder_ckpt_path).vision_model
            self.encoder.encoder.decoder_skip_layers_for_visual = [i for i in range(12)]
        else:
            raise ValueError(f'Encoder {self.args.encoder_type} is not supported')

    def _init_decoder(self):
        if self.args.decoder_type == 'd_plip':
            self.decoder = CLIPModel.from_pretrained(self.args.decoder_ckpt_path).text_model
            self.decoder_head = nn.Linear(512, 49408)
            self.decoder.encoder.decoder_skip_layers_for_visual = self.args.decoder_skip_layers_for_visual
        elif self.args.decoder_type == 'gpt2':
            self.decoder = GPT2Model.from_pretrained(self.args.decoder_ckpt_path)
            self.decoder_head = nn.Linear(768, 50258) 
            self.decoder.decoder_skip_layers_for_visual = self.args.decoder_skip_layers_for_visual
        else:
            raise ValueError(f'Decoder {self.args.decoder_type} is not supported')

    def _init_prompt(self):
        if self.args.type != 'lora':
            # Init prompt for encoder
            self.encoder_prompt = EncoderPrompt(self.args.encoder_type, 
                                                self.args.encoder_prompt_len, 
                                                self.args.encoder_skip_layers,
                                                True if self.args.type == 'distinct' else False)
            self.key, self.encoder_prompt_dict = self.encoder_prompt.prompt_combination
            if self.args.encoder_type == 'ctranspath':
                for layer_id in self.encoder_prompt_dict:
                    setattr(self.encoder, f'prompt_layer_{layer_id}', self.encoder_prompt_dict[layer_id]) 
            elif self.args.encoder_type == 'e_plip':
                for layer_id in self.encoder_prompt_dict:
                    setattr(self.encoder.encoder, f'prompt_layer_{layer_id}', self.encoder_prompt_dict[layer_id])
            # Init prompt for decoder
            self.decoder_prompt = DecoderPrompt(self.args.decoder_type, self.args.decoder_prompt_len, self.args.decoder_skip_layers)
            self.decoder_prompt_dict = self.decoder_prompt.prompt_combination[1]
            for layer_id in self.decoder_prompt_dict:
                if self.args.decoder_type == 'd_plip':
                    setattr(self.decoder.encoder, f'prompt_layer_{layer_id}', self.decoder_prompt_dict[layer_id])
                elif self.args.decoder_type == 'gpt2':
                    setattr(self.decoder, f'prompt_layer_{layer_id}', self.decoder_prompt_dict[layer_id])
        elif self.args.type == 'lora':
            self.encoder_lora = Lora(self.args, module='encoder')
            self.key, self.encoder_lora_dict = self.encoder_lora.lora_combination
            self.decoder_lora = Lora(self.args, module='decoder')
            self.decoder_lora_dict = self.decoder_lora.lora_combination[1]

            for layer_id in self.encoder_lora_dict:
                if self.args.encoder_type == 'ctranspath':
                    setattr(self.encoder, f'lora_layer_{layer_id}', self.encoder_lora_dict[layer_id])
                elif self.args.encoder_type == 'e_plip':
                    setattr(self.encoder.encoder, f'lora_layer_{layer_id}', self.encoder_lora_dict[layer_id])

            for layer_id in self.decoder_lora_dict:
                if self.args.decoder_type == 'd_plip':
                    setattr(self.decoder.encoder, f'lora_layer_{layer_id}', self.decoder_lora_dict[layer_id])
                elif self.args.decoder_type == 'gpt2':
                    setattr(self.decoder, f'lora_layer_{layer_id}', self.decoder_lora_dict[layer_id])

    def _init_tokenizer(self):
        if self.args.tokenizer_type == 'gpt2':
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.args.tokenizer_type)
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.decoder.resize_token_embeddings(len(self.tokenizer))
        else:
            self.tokenizer = AutoProcessor.from_pretrained(self.args.tokenizer_type)

    def _init_projector(self):
        self.projector = MLP(self.args)

    def _freeze_encoder_and_decoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False

    def get_query(self, img, text):
        with torch.no_grad():
            if self.args.encoder_type == 'ctranspath':
                visual_query = self.encoder(img, use_prompt=False, use_lora=False, lora_config=None)
                token = self.tokenizer(text, return_tensors="pt", padding=True)
                input_ids=token['input_ids'][:,:-1].to(self.args.device)
                attention_mask=torch.where(input_ids<50257,1,0).to(self.args.device) 
                text_query = self.decoder(input_ids=input_ids, 
                                          proj_encoder_feature=None, 
                                          attention_mask=attention_mask,
                                          use_prompt=False,
                                          use_lora=False, 
                                          lora_config=None).pooler_output
                query = torch.cat((visual_query, text_query),dim=1)
            elif self.args.encoder_type == 'e_plip':
                for layer_id in self.encoder_prompt_dict:
                    setattr(self.encoder.encoder, f'prompt_layer_{layer_id}', None)
                query = self.encoder(img)[1]
                for layer_id in self.encoder_prompt_dict:
                    setattr(self.encoder.encoder, f'prompt_layer_{layer_id}', self.encoder_prompt_dict[layer_id])
        return query

    def forward(self, img, text):
        assert len(img.shape)==4 and img.shape[1:] == (3,224,224), \
                    f'Expect img input of shape (bs,3,224,3,224) but got {img.shape}'
        # Forward through an encoder
        if self.args.encoder_type == 'ctranspath':
            img = self.encoder(img, lora_config=(self.args.lora_drop_out, self.args.lora_alpha))
        elif self.args.encoder_type == 'e_plip':
            img = self.encoder(img)[1]

        # Forward through a projector
        img = self.projector(img)
        if self.args.decoder_type == 'd_plip':
            img = img.reshape(img.shape[0], -1, 512)
        elif self.args.decoder_type == 'gpt2':
            img = img.reshape(img.shape[0], -1, 768)

        # Forward though a decoder
        text = self.tokenizer(text, return_tensors="pt", padding=True)
        input_ids = text['input_ids'].to(self.device)
        attention_mask = text['attention_mask'].to(self.device)
        output = self.decoder(proj_encoder_feature=img, 
                              input_ids=input_ids, 
                              attention_mask=attention_mask,
                              lora_config=(self.args.lora_drop_out, self.args.lora_alpha)
                              )
        logits = self.decoder_head(output.last_hidden_state)

        return {
            'input_ids': input_ids,
            'last_layer_logits': logits
        }
    
    def forward_decoder(self, proj_encoder_feature, input_ids, attention_mask):
        output = self.decoder(proj_encoder_feature=proj_encoder_feature,
                              input_ids=input_ids, 
                              attention_mask=attention_mask,
                              lora_config=(0.0, self.args.lora_alpha))
        return output

class PromptModelWithConnection(nn.Module):
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
        elif self.args.encoder_type == 'e_plip':
            self.encoder = CLIPModel.from_pretrained(self.args.decoder_ckpt_path).vision_model
            self.encoder.encoder.decoder_skip_layers_for_visual = [i for i in range(12)]
        else:
            raise ValueError(f'Encoder {args.encoder_type} is not supported')
        
        # Init projector
        self.projector = MLP(args)

        # Init text decoder
        if args.decoder_type == 'd_plip':
            self.decoder = CLIPModel.from_pretrained(args.decoder_ckpt_path).text_model
            self.decoder_head = nn.Linear(512, 49408)
            self.decoder_dim = 512
            self.decoder.encoder.decoder_skip_layers_for_visual = self.args.decoder_skip_layers_for_visual
        else:
            raise ValueError(f'Decoder {args.decoder_type} is not supported')

        # Freeze encoder and decoder
        self.freeze_encoder_and_decoder()

        # Init prompt for encoder + connection
        self.encoder_prompt = EncoderPrompt(args.encoder_type, args.encoder_prompt_len, args.encoder_skip_layers)
        self.key, self.encoder_prompt_dict = self.encoder_prompt.prompt_combination
        for layer_id in self.encoder_prompt_dict:
            setattr(self.encoder, f'prompt_layer_{layer_id}', self.encoder_prompt_dict[layer_id])
            if self.encoder_prompt_dict[layer_id] is not None:
                in_dim = self.encoder_prompt_dict[layer_id].shape[-1]
                out_dim = 512
                setattr(self, f'prompt_projector_{layer_id}', MLP_for_prompt(in_dim, out_dim))

        # Init tokenizer
        self.tokenizer = AutoProcessor.from_pretrained(args.tokenizer_type)

    def freeze_encoder_and_decoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False

    def get_query(self, x):
        with torch.no_grad():
            query = self.encoder(x)
        return query

    def forward(self, img, text):
        assert len(img.shape)==4 and img.shape[1:] == (3,224,224), \
                    f'Expect img input of shape (bs,3,224,3,224) but got {img.shape}'
        img = self.encoder(img, use_prompt=True)
        img = self.projector(img)

        # Forward prompt though connection
        for layer_id in self.encoder_prompt_dict:
            if self.encoder_prompt_dict[layer_id] is not None:
                projector = getattr(self, f'prompt_projector_{layer_id}')
                proj_prompt = projector(getattr(self.encoder, f'prompt_layer_{layer_id}'))
                setattr(self.decoder.encoder, f'prompt_layer_{layer_id}', proj_prompt)
            else:
                setattr(self.decoder.encoder, f'prompt_layer_{layer_id}', None)

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
        if True or self.args.connection:
            for layer_id in self.encoder_prompt_dict:
                if self.encoder_prompt_dict[layer_id] is not None:
                    projector = getattr(self, f'prompt_projector_{layer_id}')
                    proj_prompt = projector(getattr(self.encoder, f'prompt_layer_{layer_id}'))
                    setattr(self.decoder.encoder, f'prompt_layer_{layer_id}', proj_prompt)
                else:
                    setattr(self.decoder.encoder, f'prompt_layer_{layer_id}', None)
        output = self.decoder(proj_encoder_feature=proj_encoder_feature,
                              input_ids=input_ids, 
                              attention_mask=attention_mask)
        return output