import torch.nn as nn
import torch
from .swin_transformer import ctranspath

class SingleEncoder(nn.Module):
    def __init__(self, args, num_classes):
        super().__init__()
        self.args = args
        if args.encoder_type == 'ctranspath':
            self.model = ctranspath()
            try:
                self.model.head = nn.Linear(768, num_classes)
                td = torch.load(args.model_pth, map_location=args.device)['model_state_dict']
                for key in list(td.keys()):
                    td[key.replace('model.', '')] = td.pop(key)
                self.model.load_state_dict(td)
            except:
                self.model.head = nn.Identity()
                td = torch.load(self.args.encoder_ckpt_path, map_location=args.device)
                self.model.load_state_dict(td['model'])
                self.model.head = nn.Linear(768, num_classes)
        elif args.encoder_type == 'swin_v2_s':
            from torchvision.models import swin_v2_s, Swin_V2_S_Weights
            self.model = swin_v2_s(weights=Swin_V2_S_Weights.DEFAULT)
            self.model.head = nn.Linear(768, num_classes)
        elif args.encoder_type == 'regnet':
            from torchvision.models import regnet_x_16gf
            self.model = regnet_x_16gf(weights='DEFAULT')
            self.model.fc = nn.Linear(2048, num_classes)
        elif args.encoder_type == 'swin_v2_b':
            from torchvision.models import swin_v2_b, Swin_V2_B_Weights
            self.model = swin_v2_b(weights=Swin_V2_B_Weights.DEFAULT)
            self.model.head = nn.Linear(1024, num_classes)
        elif args.encoder_type == 'plip':
            from transformers import AutoModelForZeroShotImageClassification
            self.model = AutoModelForZeroShotImageClassification.from_pretrained("vinid/plip").vision_model
            # self.model = CLIPModel.from_pretrained(self.args.decoder_ckpt_path).vision_model
            self.model.visual_projection = nn.Linear(768, num_classes)
        elif self.args.encoder_type == 'resnet50':
            from torchvision.models import resnet50
            self.model = resnet50(weights='DEFAULT')
            self.model.fc = nn.Linear(2048, num_classes)
        elif self.args.encoder_type == 'efficientnet':
            from torchvision.models import efficientnet_v2_s
            self.model = efficientnet_v2_s(weights='DEFAULT')
            self.model.classifier[1] = nn.Linear(1280, num_classes)
        elif self.args.encoder_type == 'convnext_base':
            from torchvision.models import convnext_base               
            self.model = convnext_base(weights='DEFAULT')  
            self.model.classifier[2] = nn.Linear(1024, num_classes)
        elif self.args.encoder_type == 'vit_b_16':
            from torchvision.models import vit_b_16
            self.model = vit_b_16(weights='ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1')
            self.model.heads.head = nn.Linear(768, num_classes)
        elif self.args.encoder_type == 'maxvit':
            # import timm
            # self.model = timm.create_model('maxvit_base_tf_224', pretrained=True, num_classes=num_classes)
            from torchvision.models import maxvit_t
            self.model = maxvit_t(weights='DEFAULT')
            self.model.classifier[5] = nn.Linear(512, num_classes)
        elif self.args.encoder_type == 'resnext50':
            from torchvision.models import resnext50_32x4d
            self.model = resnext50_32x4d(weights='ResNeXt50_32X4D_Weights.IMAGENET1K_V2')
            self.model.fc = nn.Linear(2048, num_classes)
        elif self.args.encoder_type == 'swin_b':
            from torchvision.models import swin_b
            self.model = swin_b(weights='DEFAULT')
            self.model.head = nn.Linear(1024, num_classes)
            
    def forward(self, x):
        if self.args.encoder_type == 'plip':
            x = self.model(x).pooler_output
            x = self.model.visual_projection(x)
            return x
        else:
            return self.model(x)