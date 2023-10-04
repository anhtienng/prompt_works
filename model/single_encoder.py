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
        elif args.encoder_type == 'swin_v2_b':
            from torchvision.models import swin_v2_b, Swin_V2_B_Weights
            self.model = swin_v2_b(weights=Swin_V2_B_Weights.DEFAULT)
            self.model.head = nn.Linear(1024, num_classes)

    def forward(self, x):
        return self.model(x)