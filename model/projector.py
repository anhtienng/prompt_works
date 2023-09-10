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
                elif args.proj_activation == 'relu':
                    layers.append(nn.ReLU())
                else:
                    raise ValueError(f'not support {args.proj_activation}')
        self.model = nn.Sequential(*layers)


class MLP_for_prompt(nn.Module):
    def forward(self, x):
        return self.model(x)

    def __init__(self, in_dim, out_dim):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_dim, out_dim, bias=True))
        layers.append(nn.GELU())
        # layers.append(nn.Linear(768, out_dim, bias=True))
        # layers.append(nn.GELU())
        self.model = nn.Sequential(*layers)