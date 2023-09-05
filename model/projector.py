import torch.nn as nn

class MLP(nn.Module):
    def forward(self, x):
        return self.model(x)

    def __init__(self, args):
        super().__init__()
        layers = []
        sizes = args.layers_dim
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=True))
            if i < len(sizes) - 2:
                if args.proj_activation == 'gelu':
                    layers.append(nn.GELU())
                else:
                    raise ValueError(f'not support {args.proj_activation}')
        self.model = nn.Sequential(*layers)