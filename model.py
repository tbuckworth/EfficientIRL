from abc import ABC

from misc_util import orthogonal_init, xavier_uniform_init
import torch.nn as nn
import torch


class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x, start_dim=1)


class SeqModel(nn.Module):
    def __init__(self, list, device):
        super().__init__()
        self.layers = nn.Sequential(*list)
        self.device = device

    def forward(self, x):
        return self.layers(x)


class MlpModel(nn.Module):
    def __init__(self,
                 input_dims=4,
                 hidden_dims=[64, 64],
                 final_relu=True,
                 **kwargs):
        """
        input_dim:     (int)  number of the input dimensions
        hidden_dims:   (list) list of the dimensions for the hidden layers
        use_batchnorm: (bool) whether to use batchnorm
        """
        super(MlpModel, self).__init__()
        self.final_relu = final_relu
        # Hidden layers
        hidden_dims = [input_dims] + hidden_dims
        layers = []
        for i in range(len(hidden_dims) - 1):
            in_features = hidden_dims[i]
            out_features = hidden_dims[i + 1]
            layers.append(nn.Linear(in_features, out_features))
            if self.final_relu or i < len(hidden_dims) - 2:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]
        self.apply(orthogonal_init)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x



class MlpModelNoFinalRelu(nn.Module):
    def __init__(self,
                 input_dims=4,
                 hidden_dims=[64, 64],
                 **kwargs):
        """
        input_dim:     (int)  number of the input dimensions
        hidden_dims:   (list) list of the dimensions for the hidden layers
        use_batchnorm: (bool) whether to use batchnorm
        """
        super(MlpModelNoFinalRelu, self).__init__()
        self.embedder = lambda x: x
        self.output_dim = hidden_dims[-1]
        if isinstance(self.output_dim, list):
            hidden_dims = hidden_dims[:-1]
            self.final_layers = nn.ModuleList([orthogonal_init(nn.Linear(hidden_dims[-1], k)) for k in self.output_dim])
        else:
            self.final_layers = None

        # Hidden layers
        hidden_dims = [input_dims] + hidden_dims
        layers = []

        for i in range(len(hidden_dims) - 1):
            in_features = hidden_dims[i]
            out_features = hidden_dims[i + 1]
            layers.append(nn.Linear(in_features, out_features))
            if i < len(hidden_dims) - 2:
                layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)
        self.apply(orthogonal_init)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        if self.final_layers is None:
            return x
        return (fl(x) for fl in self.final_layers)

    def embed_and_forward(self, obs, action=None):
        x = self.embedder(obs)
        if action is not None:
            x = torch.concat((x, action.unsqueeze(-1)), dim=-1)
        for layer in self.layers:
            x = layer(x)
        return x


class NatureModel(nn.Module):
    def __init__(self,
                 in_channels,
                 **kwargs):
        """
        input_shape:  (tuple) tuple of the input dimension shape (channel, height, width)
        filters:       (list) list of the tuples consists of (number of channels, kernel size, and strides)
        use_batchnorm: (bool) whether to use batchnorm
        """
        super(NatureModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1), nn.ReLU(),
            Flatten(),
            nn.Linear(in_features=64 * 7 * 7, out_features=512), nn.ReLU()
        )
        self.output_dim = 512
        self.apply(orthogonal_init)

    def forward(self, x):
        x = self.layers(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = nn.ReLU()(x)
        out = self.conv1(out)
        out = nn.ReLU()(out)
        out = self.conv2(out)
        return out + x


class ImpalaBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ImpalaBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


scale = 1

class ImpalaModel(nn.Module):
    def __init__(self,
                 in_channels,
                 **kwargs):
        super(ImpalaModel, self).__init__()
        self.block1 = ImpalaBlock(in_channels=in_channels, out_channels=16 * scale)
        self.block2 = ImpalaBlock(in_channels=16 * scale, out_channels=32 * scale)
        self.block3 = ImpalaBlock(in_channels=32 * scale, out_channels=32 * scale)
        self.fc = nn.Linear(in_features=32 * scale * 8 * 8, out_features=256)

        self.output_dim = 256
        self.apply(xavier_uniform_init)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = nn.ReLU()(x)
        x = Flatten()(x)
        x = self.fc(x)
        x = nn.ReLU()(x)
        return x




class GRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRU, self).__init__()
        self.gru = orthogonal_init(nn.GRU(input_size, hidden_size), gain=1.0)

    def forward(self, x, hxs, masks):
        # Prediction
        if x.size(0) == hxs.size(0):
            # input for GRU-CELL: (L=sequence_length, N, H)
            # output for GRU-CELL: (output: (L, N, H), hidden: (L, N, H))
            masks = masks.unsqueeze(-1)
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        # Training
        # We will recompute the hidden state to allow gradient to be back-propagated through time
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            # (can be interpreted as a truncated back-propagation through time)
            has_zeros = ((masks[1:] == 0.0) \
                         .any(dim=-1)
                         .nonzero()
                         .squeeze()
                         .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class IdentityModel(nn.Module):
    def __init__(self):
        super(IdentityModel, self).__init__()

    def forward(self, x):
        return x

class MLPLayers(nn.Module, ABC):
    def __init__(self,*args, **kwargs):
        super(MLPLayers, self).__init__(*args, **kwargs)

    def generate_layers(self, input_dims, hidden_dims, output_dim):
        hidden_dims = [input_dims] + hidden_dims
        layers = []
        for i in range(len(hidden_dims) - 1):
            in_features = hidden_dims[i]
            out_features = hidden_dims[i + 1]
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        return layers


class NextRewModel(MLPLayers):
    def __init__(self,
                 input_dims,
                 hidden_dims,
                 action_size,
                 device
                 ):
        """
        input_dim:      (int)  number of the input dimensions
        hidden_dims:    (list) list of the dimensions for the hidden layers
        output_dim:     (int)
        action_size:    (int)
        device:         (str)
        """
        super(NextRewModel, self).__init__()
        self.action_size = action_size
        self.device = device
        self.output_dim = 1
        # Hidden layers
        # hidden_dims = [input_dims] + hidden_dims
        # layers = []
        # for i in range(len(hidden_dims) - 1):
        #     in_features = hidden_dims[i]
        #     out_features = hidden_dims[i + 1]
        #     layers.append(nn.Linear(in_features, out_features))
        #     layers.append(nn.ReLU())
        # layers.append(nn.Linear(hidden_dims[-1], output_dim))
        # self.layers = nn.Sequential(*layers)
        self.layers = nn.Sequential(*self.generate_layers(input_dims, hidden_dims, self.output_dim))
        self.apply(orthogonal_init)
        self.to(self.device)

    def forward(self, h, a):
        a_hot = torch.nn.functional.one_hot(a.to(torch.int64)).to(device=self.device).float()
        x = torch.concat((h, a_hot), dim=-1)
        for layer in self.layers:
            x = layer(x)
        return x


class RewValModel(MLPLayers):
    def __init__(self,
                 input_dims,
                 hidden_dims,
                 device
                 ):
        """
        input_dim:     (int)  number of the input dimensions
        hidden_dims:   (list) list of the dimensions for the hidden layers
        device:
        """
        super(RewValModel, self).__init__()
        self.device = device
        self.output_dim = 1
        # Hidden layers
        self.rew_layers = nn.Sequential(*self.generate_layers(input_dims, hidden_dims, self.output_dim))
        self.val_layers = nn.Sequential(*self.generate_layers(input_dims, hidden_dims, self.output_dim))
        self.apply(orthogonal_init)
        self.to(self.device)

    # def generate_layers(self, input_dims, hidden_dims, output_dim):
    #     hidden_dims = [input_dims] + hidden_dims
    #     layers = []
    #     for i in range(len(hidden_dims) - 1):
    #         in_features = hidden_dims[i]
    #         out_features = hidden_dims[i + 1]
    #         layers.append(nn.Linear(in_features, out_features))
    #         layers.append(nn.ReLU())
    #     layers.append(nn.Linear(hidden_dims[-1], output_dim))
    #     return layers

    def forward(self, h):
        return self.reward(h), self.value(h)

    def reward(self, x):
        for layer in self.rew_layers:
            x = layer(x)
        return x

    def value(self, x):
        for layer in self.val_layers:
            x = layer(x)
        return x


class ImpalaValueModel(MLPLayers):
    def __init__(self,
                 in_channels,
                 hidden_dims,
                 output_dim=1,
                 **kwargs):
        super(ImpalaValueModel, self).__init__()
        self.model = ImpalaModel(in_channels=in_channels, **kwargs)
        self.output_dim = output_dim
        if isinstance(output_dim, int):
            self.layers = nn.Sequential(*self.generate_layers(self.model.output_dim, hidden_dims, self.output_dim))
            self.final_layers = None
        elif isinstance(output_dim, list):
            self.layers = nn.Sequential(*(self.generate_layers(self.model.output_dim, hidden_dims[:-1], hidden_dims[-1]) + [nn.ReLU()]))
            self.final_layers = nn.ModuleList([orthogonal_init(nn.Linear(hidden_dims[-1], k)) for k in output_dim])
        else:
            raise NotImplementedError("output_dim must be int or list")

        self.apply(xavier_uniform_init)

    def forward(self, x):
        h = self.model(x)
        for layer in self.layers:
            h = layer(h)
        if self.final_layers is None:
            return h
        return (fl(h) for fl in self.final_layers)
