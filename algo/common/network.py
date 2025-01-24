import logging
import math
from copy import deepcopy
from typing import Mapping, Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


def orthogonal_init(layer, gain=np.sqrt(2)):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)


class BaseLSTM(nn.Module):
    def init_hidden_state(self, device=torch.device('cpu'), batch_size=None):
        if batch_size is None:
            self.hidden_state = torch.zeros(self.lstm.num_layers, self.lstm.hidden_size).to(device)
            self.cell_state = torch.zeros(self.lstm.num_layers, self.lstm.hidden_size).to(device)
        else:
            self.hidden_state = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device)
            self.cell_state = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device)

    def forward(self, s):
        raise NotImplementedError


class Actor_LSTM(BaseLSTM):
    def __init__(self, args):
        super(Actor_LSTM, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(args.state_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.lstm_hidden_dim),
        )

        self.lstm = nn.LSTM(input_size=args.lstm_hidden_dim, hidden_size=args.lstm_hidden_dim,
                            num_layers=args.lstm_num_layers, batch_first=True)

        self.mean_layer = nn.Linear(args.lstm_hidden_dim, args.action_dim)

        if args.std:
            self.log_std = nn.Parameter(torch.tensor(args.std).log(), requires_grad=False)
        else:
            self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))

        if args.use_orthogonal_init:
            logging.info("------use_orthogonal_init------")
            if hasattr(self, 'mean_layer'):
                orthogonal_init(self.mean_layer, gain=0.01)

            if hasattr(self, 'std_layer'):
                orthogonal_init(self.std_layer, gain=0.01)

    def forward(self, s):
        s_b = s['base']
        # s_b: [batch_size, seq_len, arg.state_dim]

        # s_p = s['privilege']
        s_b = self.fc1(s_b)
        # s_p: [batch_size, seq_len, hidden_dim]

        s, (self.hidden_state, self.cell_state) = self.lstm(s_b, (self.hidden_state, self.cell_state))
        # s: [batch_size, seq_len, hidden_size_lstm]

        mean = torch.tanh(self.mean_layer(s))
        # mean: [batch_size, seq_len, action_dim]

        return mean, self.log_std.expand_as(mean).exp()

    def pdf(self, s):
        mean, std = self.forward(s)
        return Normal(mean, std)


class Critic_LSTM(BaseLSTM):
    def __init__(self, args):
        super(Critic_LSTM, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(args.privilege_state_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.lstm_hidden_dim),
        )

        self.lstm = nn.LSTM(input_size=args.lstm_hidden_dim, hidden_size=args.lstm_hidden_dim,
                            num_layers=args.lstm_num_layers, batch_first=True)

        self.value_layer = nn.Sequential(
            nn.Linear(args.lstm_hidden_dim, args.lstm_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.lstm_hidden_dim, 1)
        )

    def forward(self, s):
        s_p = s['privilege']
        # s_p: [batch_size, seq_len, hidden_dim]

        s_p = self.fc1(s_p)
        # s_p: [batch_size, seq_len, hidden_dim]

        s, (self.hidden_state, self.cell_state) = self.lstm(s_p, (self.hidden_state, self.cell_state))
        # s: [batch_size, seq_len, hidden_size_lstm]

        value = self.value_layer(s)
        # mean: [batch_size, seq_len, action_dim]

        return value


class Actor_LSTM_v2(BaseLSTM):
    def __init__(self, args):
        super(Actor_LSTM_v2, self).__init__()

        state_dim = args.state_dim

        if 'hfield' in state_dim:
            del state_dim['hfield']

        state_dim = sum([math.prod(shape) for shape in state_dim.values()])

        self.lstm = nn.LSTM(input_size=state_dim, hidden_size=args.lstm_hidden_dim,
                            num_layers=args.lstm_num_layers, batch_first=True)

        self.mean_layer = nn.Linear(args.lstm_hidden_dim, args.action_dim)

        if args.std:
            self.log_std = nn.Parameter(torch.tensor(args.std).log(), requires_grad=False)
        else:
            self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))

        if args.use_orthogonal_init:
            logging.info("------use_orthogonal_init------")
            if hasattr(self, 'mean_layer'):
                orthogonal_init(self.mean_layer, gain=0.01)

            if hasattr(self, 'std_layer'):
                orthogonal_init(self.std_layer, gain=0.01)

    # def forward(self, s):
    #     s = torch.cat(list(s.values()), dim=-1)
    #     # s: [batch_size, seq_len, *]
    #
    #     s, (self.hidden_state, self.cell_state) = self.lstm(s, (self.hidden_state, self.cell_state))
    #     # s: [batch_size, seq_len, hidden_size_lstm]
    #
    #     mean = self.mean_layer(s)
    #     # mean: [batch_size, seq_len, action_dim]
    #
    #     return mean, self.log_std.expand_as(mean).exp()

    def forward(self, s):
        s = torch.cat([v for k, v in s.items() if k != 'hfield'], dim=-1)
        # s: [E, T, N, *F]

        E, T, N, *F = s.size()

        s = s.transpose(1, 2)
        # s: [E, N, T, *F]

        s = s.reshape(E * N, T, -1)
        # s: [E * N, T, prod(*F)]

        s, (self.hidden_state, self.cell_state) = self.lstm(s, (self.hidden_state, self.cell_state))
        # s: [E * N, T, H]

        s = s.view(E, N, T, -1)
        # s: [E, N, T, H]

        s = s.transpose(1, 2)
        # s: [E, T, N, H]

        mean = self.mean_layer(s)
        # mean: [E, T, N, action_dim]

        return mean, self.log_std.expand_as(mean).exp()

    def pdf(self, s):
        mean, std = self.forward(s)
        return Normal(mean, std)


class Critic_LSTM_v2(BaseLSTM):
    def __init__(self, args):
        super(Critic_LSTM_v2, self).__init__()

        state_dim = args.state_dim

        if 'hfield' in state_dim:
            del state_dim['hfield']

        state_dim = sum([math.prod(shape) for shape in state_dim.values()])

        self.lstm = nn.LSTM(input_size=state_dim, hidden_size=args.lstm_hidden_dim,
                            num_layers=args.lstm_num_layers, batch_first=True)

        self.value_layer = nn.Sequential(
            nn.Linear(args.lstm_hidden_dim, 1)
        )

    def forward(self, s):
        s = torch.cat([v for k, v in s.items() if k != 'hfield'], dim=-1)
        # s: [E, T, N, *F]

        E, T, N, *F = s.size()

        s = s.transpose(1, 2)
        # s: [E, N, T, *F]

        s = s.reshape(E * N, T, -1)
        # s: [E * N, T, prod(*F)]

        s, (self.hidden_state, self.cell_state) = self.lstm(s, (self.hidden_state, self.cell_state))
        # s: [E * N, T, H]

        s = s.view(E, N, T, -1)
        # s: [E, N, T, H]

        s = s.transpose(1, 2)
        # s: [E, T, N, H]

        value = self.value_layer(s)
        # mean: [E, T, N, 1]

        return value


class Actor_LSTM_v3(BaseLSTM):
    def __init__(self, args):
        super(Actor_LSTM_v3, self).__init__()

        self.fc1 = nn.Linear(11, 32)

        self.lstm = nn.LSTM(input_size=args.state_dim + self.fc1.out_features - 6, hidden_size=args.lstm_hidden_dim,
                            num_layers=args.lstm_num_layers, batch_first=True)

        self.mean_layer = nn.Linear(args.lstm_hidden_dim, args.action_dim)

        if args.std:
            self.log_std = nn.Parameter(torch.tensor(args.std).log(), requires_grad=False)
        else:
            self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))

        if args.use_orthogonal_init:
            logging.info("------use_orthogonal_init------")
            if hasattr(self, 'mean_layer'):
                orthogonal_init(self.mean_layer, gain=0.01)

            if hasattr(self, 'std_layer'):
                orthogonal_init(self.std_layer, gain=0.01)

    def forward(self, s):
        s = s['base']
        # s: [batch_size, seq_len, hidden_dim]

        s_base_orient_base_yaw = s[..., :5]
        # s_base_orient_base_yaw: [batch_size, seq_len, 5]

        s_motor = s[..., :-6]
        # s_motor: [batch_size, seq_len, arg.state_dim-6]

        s_cmd_encoding = s[..., -6:]
        # s_cmd_encoding: [batch_size, seq_len, 6]

        s_latent = torch.cat((s_base_orient_base_yaw, s_cmd_encoding), dim=-1)
        # s_latent: [batch_size, seq_len, 11]

        s_latent = self.fc1(s_latent)
        # s_cmd_encoding: [batch_size, seq_len, 32]

        s = torch.cat((s_motor, s_latent), dim=-1)
        # s: [batch_size, seq_len, arg.state_dim + 32 - 6]

        s, (self.hidden_state, self.cell_state) = self.lstm(s, (self.hidden_state, self.cell_state))
        # s: [batch_size, seq_len, hidden_size_lstm]

        mean = self.mean_layer(s)
        # mean: [batch_size, seq_len, action_dim]

        return mean, self.log_std.expand_as(mean).exp()

    def pdf(self, s):
        mean, std = self.forward(s)
        return Normal(mean, std)


class Critic_LSTM_v3(BaseLSTM):
    def __init__(self, args):
        super(Critic_LSTM_v3, self).__init__()

        self.fc1 = nn.Linear(11, 32)

        self.lstm = nn.LSTM(input_size=args.privilege_state_dim + self.fc1.out_features - 6,
                            hidden_size=args.lstm_hidden_dim,
                            num_layers=args.lstm_num_layers, batch_first=True)

        self.value_layer = nn.Sequential(
            nn.Linear(args.lstm_hidden_dim, 1)
        )

    def forward(self, s):
        s = s['privilege']
        # s: [batch_size, seq_len, hidden_dim]

        s_base_orient_base_yaw = s[..., :5]
        # s_base_orient_base_yaw: [batch_size, seq_len, 5]

        s_motor = s[..., :-6]
        # s_motor: [batch_size, seq_len, arg.state_dim-6]

        s_cmd_encoding = s[..., -6:]
        # s_cmd_encoding: [batch_size, seq_len, 6]

        s_latent = torch.cat((s_base_orient_base_yaw, s_cmd_encoding), dim=-1)
        # s_latent: [batch_size, seq_len, 11]

        s_latent = self.fc1(s_latent)
        # s_cmd_encoding: [batch_size, seq_len, 32]

        s = torch.cat((s_motor, s_latent), dim=-1)
        # s: [batch_size, seq_len, arg.state_dim - 6 + 32]

        s, (self.hidden_state, self.cell_state) = self.lstm(s, (self.hidden_state, self.cell_state))
        # s: [batch_size, seq_len, hidden_size_lstm]

        value = self.value_layer(s)
        # mean: [batch_size, seq_len, action_dim]

        return value


class Actor_FF(nn.Module):
    def __init__(self, args):
        super(Actor_FF, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(args.state_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.action_dim),
        )

        if args.std:
            self.log_std = nn.Parameter(torch.tensor(args.std).log(), requires_grad=False)
        else:
            self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))

        if args.use_orthogonal_init:
            logging.info("------use_orthogonal_init------")
            if hasattr(self, 'fc1'):
                orthogonal_init(self.fc1, gain=0.01)

            if hasattr(self, 'std_layer'):
                orthogonal_init(self.std_layer, gain=0.01)

    def forward(self, s):
        s_b = s['base']
        # s_b: [batch_size, seq_len, arg.state_dim]

        mean = self.fc1(s_b)
        # mean: [batch_size, seq_len, action_dim]

        return mean, self.log_std.expand_as(mean).exp()

    def pdf(self, s):
        mean, std = self.forward(s)
        return Normal(mean, std)


class Critic_FF(nn.Module):
    def __init__(self, args):
        super(Critic_FF, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(args.privilege_state_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, 1),
        )

        if args.use_orthogonal_init:
            logging.info("------use_orthogonal_init------")
            if hasattr(self, 'fc1'):
                orthogonal_init(self.fc1, gain=0.01)

    def forward(self, s):
        s_b = s['privilege']
        # s_b: [batch_size, seq_len, arg.state_dim]

        value = self.fc1(s_b)
        # mean: [batch_size, seq_len, 1]

        return value


class Actor_FF_v2(nn.Module):
    def __init__(self, args):
        super(Actor_FF_v2, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(args.state_dim, args.hidden_dim),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.Linear(args.hidden_dim, args.action_dim),
        )

        if args.std:
            self.log_std = nn.Parameter(torch.tensor(args.std).log(), requires_grad=False)
        else:
            self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))

        if args.use_orthogonal_init:
            logging.info("------use_orthogonal_init------")
            if hasattr(self, 'fc1'):
                orthogonal_init(self.fc1, gain=0.01)

            if hasattr(self, 'std_layer'):
                orthogonal_init(self.std_layer, gain=0.01)

    def forward(self, s):
        s_b = s['base']
        # s_b: [batch_size, seq_len, arg.state_dim]

        mean = self.fc1(s_b)
        # mean: [batch_size, seq_len, action_dim]

        return mean, self.log_std.expand_as(mean).exp()

    def pdf(self, s):
        mean, std = self.forward(s)
        return Normal(mean, std)


class Critic_FF_v2(nn.Module):
    def __init__(self, args):
        super(Critic_FF_v2, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(args.privilege_state_dim, args.hidden_dim),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.Linear(args.hidden_dim, 1),
        )

        if args.use_orthogonal_init:
            logging.info("------use_orthogonal_init------")
            if hasattr(self, 'fc1'):
                orthogonal_init(self.fc1, gain=0.01)

    def forward(self, s):
        s_b = s['privilege']
        # s_b: [batch_size, seq_len, arg.state_dim]

        value = self.fc1(s_b)
        # mean: [batch_size, seq_len, 1]

        return value


class Actor_FF_v3(nn.Module):
    def __init__(self, args):
        super(Actor_FF_v3, self).__init__()

        state_dim = sum([math.prod(shape) for shape in args.state_dim.values()])

        self.fc1 = nn.Linear(state_dim, args.action_dim)

        if args.std:
            self.log_std = nn.Parameter(torch.tensor(args.std).log(), requires_grad=False)
        else:
            self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))

        if args.use_orthogonal_init:
            logging.info("------use_orthogonal_init------")
            if hasattr(self, 'fc1'):
                orthogonal_init(self.fc1, gain=0.01)

            if hasattr(self, 'std_layer'):
                orthogonal_init(self.std_layer, gain=0.01)

    def forward(self, s):
        s = torch.cat(list(s.values()), dim=-1)
        # s: [E, T, N, *F]
        E, T, N, *F = s.size()

        s = s.view(E, T, N, -1)
        # s: [E, T, N, F]

        mean = self.fc1(s)
        # mean: [E, T, N, action_dim]

        return mean, self.log_std.expand_as(mean).exp()

    def pdf(self, s):
        mean, std = self.forward(s)
        return Normal(mean, std)


class Critic_FF_v3(nn.Module):
    def __init__(self, args):
        super(Critic_FF_v3, self).__init__()

        state_dim = sum([math.prod(shape) for shape in args.state_dim.values()])

        self.fc1 = nn.Linear(state_dim, 1)

        if args.use_orthogonal_init:
            logging.info("------use_orthogonal_init------")
            if hasattr(self, 'fc1'):
                orthogonal_init(self.fc1, gain=0.01)

    def forward(self, s):
        s = torch.cat(list(s.values()), dim=-1)
        # s: [E, T, N, *F]

        E, T, N, *F = s.size()

        s = s.view(E, T, N, -1)
        # s: [E, T, N, F]

        value = self.fc1(s)
        # mean: [E, T, N, 1]

        return value


class Actor_FF_v4(nn.Module):
    def __init__(self, args):
        super(Actor_FF_v4, self).__init__()

        state_dim = sum([math.prod(shape) for shape in args.state_dim.values()])

        self.fc1 = nn.Sequential(
            nn.Linear(state_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.action_dim),
        )

        if args.std:
            self.log_std = nn.Parameter(torch.tensor(args.std).log(), requires_grad=False)
        else:
            self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))

        if args.use_orthogonal_init:
            logging.info("------use_orthogonal_init------")
            if hasattr(self, 'fc1'):
                orthogonal_init(self.fc1, gain=0.01)

            if hasattr(self, 'std_layer'):
                orthogonal_init(self.std_layer, gain=0.01)

    def forward(self, s):
        s = torch.cat(list(s.values()), dim=-1)
        # s: [E, T, N, *F]

        E, T, N, *F = s.size()

        s = s.view(E, T, N, -1)
        # s: [E, T, N, F]

        mean = self.fc1(s)
        # mean: [E, T, N, action_dim]

        return mean, self.log_std.expand_as(mean).exp()

    def pdf(self, s):
        mean, std = self.forward(s)
        return Normal(mean, std)


class Critic_FF_v4(nn.Module):
    def __init__(self, args):
        super(Critic_FF_v4, self).__init__()

        state_dim = sum([math.prod(shape) for shape in args.state_dim.values()])

        self.fc1 = nn.Sequential(
            nn.Linear(state_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, 1),
        )

        if args.use_orthogonal_init:
            logging.info("------use_orthogonal_init------")
            if hasattr(self, 'fc1'):
                orthogonal_init(self.fc1, gain=0.01)

    def forward(self, s):
        s = torch.cat(list(s.values()), dim=-1)
        # s: [E, T, N, *F]

        E, T, N, *F = s.size()

        s = s.view(E, T, N, -1)
        # s: [E, T, N, F]

        value = self.fc1(s)
        # mean: [E, T, N, 1]

        return value


class Actor_FF_v5(nn.Module):
    def __init__(self, args):
        super(Actor_FF_v5, self).__init__()

        state_dim = sum([math.prod(shape) for shape in args.state_dim.values()])

        self.fc1 = nn.Sequential(
            nn.Linear(state_dim, args.hidden_dim),
            nn.Linear(args.hidden_dim, args.action_dim),
        )

        if args.std:
            self.log_std = nn.Parameter(torch.tensor(args.std).log(), requires_grad=False)
        else:
            self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))

        if args.use_orthogonal_init:
            logging.info("------use_orthogonal_init------")
            if hasattr(self, 'fc1'):
                orthogonal_init(self.fc1, gain=0.01)

            if hasattr(self, 'std_layer'):
                orthogonal_init(self.std_layer, gain=0.01)

    def forward(self, s):
        s = torch.cat(list(s.values()), dim=-1)
        # s: [E, T, N, *F]

        E, T, N, *F = s.size()

        s = s.view(E, T, N, -1)
        # s: [E, T, N, F]

        mean = self.fc1(s)
        # mean: [E, T, N, action_dim]

        return mean, self.log_std.expand_as(mean).exp()

    def pdf(self, s):
        mean, std = self.forward(s)
        return Normal(mean, std)


class Critic_FF_v5(nn.Module):
    def __init__(self, args):
        super(Critic_FF_v5, self).__init__()

        state_dim = sum([math.prod(shape) for shape in args.state_dim.values()])

        self.fc1 = nn.Sequential(
            nn.Linear(state_dim, args.hidden_dim),
            nn.Linear(args.hidden_dim, 1),
        )

        if args.use_orthogonal_init:
            logging.info("------use_orthogonal_init------")
            if hasattr(self, 'fc1'):
                orthogonal_init(self.fc1, gain=0.01)

    def forward(self, s):
        s = torch.cat(list(s.values()), dim=-1)
        # s: [E, T, N, *F]

        E, T, N, *F = s.size()

        s = s.view(E, T, N, -1)
        # s: [E, T, N, F]

        value = self.fc1(s)
        # mean: [E, T, N, 1]

        return value


# Using lstm for temporal feature
class BaseTransformer(nn.Module):
    def __init__(self, args):
        super(BaseTransformer, self).__init__()

        state_dim = sum([math.prod(shape) for shape in args.state_dim.values()])

        self.lstm = nn.LSTM(input_size=state_dim, hidden_size=args.lstm_hidden_dim,
                            num_layers=args.lstm_num_layers, batch_first=True)

        self.use_transformer = False

        self.is_decentralized = False

        if self.use_transformer:
            self.fc1 = nn.Sequential(
                nn.Linear(args.lstm_hidden_dim, args.transformer_hidden_dim),
                nn.ReLU(),
            )

            self.encoder_layer = nn.TransformerEncoderLayer(d_model=args.transformer_hidden_dim,
                                                            nhead=args.transformer_num_heads,
                                                            batch_first=True)

            self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=args.transformer_num_layers)

    def init_hidden_state(self, device=torch.device('cpu'), batch_size=None):
        if batch_size is None:
            self.hidden_state = torch.zeros(self.lstm.num_layers, self.lstm.hidden_size).to(device)
            self.cell_state = torch.zeros(self.lstm.num_layers, self.lstm.hidden_size).to(device)
        else:
            self.hidden_state = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device)
            self.cell_state = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device)

    def _forward_latent(self, s, src_key_padding_mask, need_weights=False):
        s = torch.cat(list(s.values()), dim=-1)
        # s: [E, T, N, *F]

        E, T, N, *F = s.size()

        s = s.transpose(1, 2)
        # s: [E, N, T, *F]

        s = s.reshape(E * N, T, -1)
        # s: [E * N, T, prod(*F)]

        s, (self.hidden_state, self.cell_state) = self.lstm(s, (self.hidden_state, self.cell_state))
        # s: [E * N, T, H=hidden_size_lstm]

        if self.use_transformer:
            s = self.fc1(s)
            # s: [E * N, T, H=transformer_hidden_dim]

        s = s.view(E, N, T, -1)
        # s: [E, N, T, H]

        s = s.transpose(1, 2)
        # s: [E, T, N, H]

        if self.use_transformer:
            s = s.reshape(E * T, N, -1)
            # s: [E * T, N, H]

            src_key_padding_mask = src_key_padding_mask.view(E * T, N)
            # src_key_padding_mask: [E * T, N]

            active_seq_mask = (~src_key_padding_mask).any(-1)
            # active_seq_idx: [E * T]

            # Prevents in-place modification for computed gradients
            s_out = s.clone()

            if self.is_decentralized:
                mask = ~torch.eye(N, dtype=torch.bool, device=s.device)
            else:
                mask = torch.zeros(N, N, dtype=torch.bool, device=s.device)

            if need_weights:
                s_out[active_seq_mask], attn_weight = self.transformer_encoder(s[active_seq_mask],
                                                                               mask=mask,
                                                                               src_key_padding_mask=
                                                                               src_key_padding_mask[
                                                                                   active_seq_mask],
                                                                               need_weights=need_weights)
            else:
                s_out[active_seq_mask] = self.transformer_encoder(s[active_seq_mask],
                                                                  mask=mask,
                                                                  src_key_padding_mask=src_key_padding_mask[
                                                                      active_seq_mask])
            s = s_out
            # s: [E * T, N, H]

            s = s.view(E, T, N, -1)
            # s: [E, T, N, H]

        if need_weights:
            return s, attn_weight

        return s

    def forward(self, s, src_key_padding_mask, need_weights=False):
        raise NotImplementedError


class Actor_Transformer(BaseTransformer):
    def __init__(self, args):
        super(Actor_Transformer, self).__init__(args)

        self.mean_layer = nn.Linear(args.transformer_hidden_dim, args.action_dim)

        if args.std:
            self.log_std = nn.Parameter(torch.tensor(args.std).log(), requires_grad=False)
        else:
            self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))

        if args.use_orthogonal_init:
            if hasattr(self, 'mean_layer'):
                orthogonal_init(self.mean_layer, gain=0.01)

            if hasattr(self, 'std_layer'):
                orthogonal_init(self.std_layer, gain=0.01)

    # s: [E=num_episodes/batch_size, T=seq_len, N=num_cassie, *F=*num_features]
    def forward(self, s, src_key_padding_mask, need_weights=False):
        if need_weights:
            s, attn_weight = self._forward_latent(s, src_key_padding_mask, need_weights)
        else:
            s = self._forward_latent(s, src_key_padding_mask, need_weights)
            # s: [E, T, N, H]

        mean = self.mean_layer(s)
        # mean: [E, T, N, action_dim]

        if need_weights:
            return mean, self.log_std.expand_as(mean).exp(), attn_weight

        return mean, self.log_std.expand_as(mean).exp()

    def pdf(self, s, src_key_padding_mask, need_weights=False):

        if need_weights:
            mean, std, attn_weight = self.forward(s, src_key_padding_mask, need_weights)
            return Normal(mean, std), attn_weight

        mean, std = self.forward(s, src_key_padding_mask, need_weights)
        return Normal(mean, std)


class Critic_Transformer(BaseTransformer):
    def __init__(self, args):
        super(Critic_Transformer, self).__init__(args)

        self.value_layer = nn.Linear(args.transformer_hidden_dim, 1)

        if args.use_orthogonal_init:
            if hasattr(self, 'value_layer'):
                orthogonal_init(self.value_layer, gain=0.01)

    def forward(self, s, src_key_padding_mask, need_weights=False):
        s = self._forward_latent(s, src_key_padding_mask, need_weights)
        # s: [E, T, N, H]

        if need_weights:
            value, attn_weight = self.value_layer(s)
            # value: [E, T, N, 1]
            return value, attn_weight

        value = self.value_layer(s)

        return value


# Using ff for temporal feature
class BaseTransformer_v2(nn.Module):
    def __init__(self, args):
        super(BaseTransformer_v2, self).__init__()

        state_dim = sum([math.prod(shape) for shape in args.state_dim.values()])

        # self.fc1 = nn.Linear(in_features=state_dim, out_features=args.lstm_hidden_dim)
        self.fc1 = nn.Sequential(
            nn.Linear(state_dim, args.lstm_hidden_dim),
            nn.ReLU(),
        )

        self.use_transformer = False

        self.is_decentralized = False

        if self.use_transformer:
            self.fc2 = nn.Sequential(
                nn.Linear(args.lstm_hidden_dim, args.transformer_hidden_dim),
                nn.ReLU(),
            )

            self.encoder_layer = nn.TransformerEncoderLayer(d_model=args.transformer_hidden_dim,
                                                            nhead=args.transformer_num_heads,
                                                            dim_feedforward=args.transformer_dim_feedforward,
                                                            batch_first=True)

            self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=args.transformer_num_layers)

    def _forward_latent(self, s, src_key_padding_mask, need_weights=False):
        s = torch.cat(list(s.values()), dim=-1)
        # s: [E, T, N, *F]

        E, T, N, *F = s.size()

        s = s.transpose(1, 2)
        # s: [E, N, T, *F]

        s = s.reshape(E * N, T, -1)
        # s: [E * N, T, prod(*F)]

        s = self.fc1(s)
        # s: [E * N, T, H=hidden_size_lstm]

        if self.use_transformer:
            s = self.fc2(s)
            # s: [E * N, T, H=transformer_hidden_dim]

        s = s.view(E, N, T, -1)
        # s: [E, N, T, H]

        s = s.transpose(1, 2)
        # s: [E, T, N, H]

        if self.use_transformer:
            s = s.reshape(E * T, N, -1)
            # s: [E * T, N, H]

            src_key_padding_mask = src_key_padding_mask.view(E * T, N)
            # src_key_padding_mask: [E * T, N]

            active_seq_mask = (~src_key_padding_mask).any(-1)
            # active_seq_idx: [E * T]

            # Prevents in-place modification for computed gradients
            s_out = s.clone()

            if self.is_decentralized:
                mask = ~torch.eye(N, dtype=torch.bool, device=s.device)
            else:
                mask = torch.zeros(N, N, dtype=torch.bool, device=s.device)

            if need_weights:
                s_out[active_seq_mask], attn_weight = self.transformer_encoder(s[active_seq_mask],
                                                                               mask=mask,
                                                                               src_key_padding_mask=
                                                                               src_key_padding_mask[
                                                                                   active_seq_mask],
                                                                               need_weights=need_weights)
            else:
                s_out[active_seq_mask] = self.transformer_encoder(s[active_seq_mask],
                                                                  mask=mask,
                                                                  src_key_padding_mask=src_key_padding_mask[
                                                                      active_seq_mask])
            s = s_out
            # s: [E * T, N, H]

            s = s.view(E, T, N, -1)
            # s: [E, T, N, H]

        if need_weights:
            return s, attn_weight

        return s

    def forward(self, s, src_key_padding_mask, need_weights=False):
        raise NotImplementedError


class Actor_Transformer_v2(BaseTransformer_v2):
    def __init__(self, args):
        super(Actor_Transformer_v2, self).__init__(args)

        self.mean_layer = nn.Linear(args.transformer_hidden_dim, args.action_dim)

        if args.std:
            self.log_std = nn.Parameter(torch.tensor(args.std).log(), requires_grad=False)
        else:
            self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))

        if args.use_orthogonal_init:
            if hasattr(self, 'mean_layer'):
                orthogonal_init(self.mean_layer, gain=0.01)

            if hasattr(self, 'std_layer'):
                orthogonal_init(self.std_layer, gain=0.01)

    # s: [E=num_episodes/batch_size, T=seq_len, N=num_cassie, *F=*num_features]
    def forward(self, s, src_key_padding_mask, need_weights=False):
        if need_weights:
            s, attn_weight = self._forward_latent(s, src_key_padding_mask, need_weights)
        else:
            s = self._forward_latent(s, src_key_padding_mask, need_weights)
            # s: [E, T, N, H]

        mean = self.mean_layer(s)
        # mean: [E, T, N, action_dim]

        if need_weights:
            return mean, self.log_std.expand_as(mean).exp(), attn_weight

        return mean, self.log_std.expand_as(mean).exp()

    def pdf(self, s, src_key_padding_mask, need_weights=False):

        if need_weights:
            mean, std, attn_weight = self.forward(s, src_key_padding_mask, need_weights)
            return Normal(mean, std), attn_weight

        mean, std = self.forward(s, src_key_padding_mask, need_weights)
        return Normal(mean, std)


class Critic_Transformer_v2(BaseTransformer_v2):
    def __init__(self, args):
        super(Critic_Transformer_v2, self).__init__(args)

        self.value_layer = nn.Linear(args.transformer_hidden_dim, 1)

        if args.use_orthogonal_init:
            if hasattr(self, 'value_layer'):
                orthogonal_init(self.value_layer, gain=0.01)

    def forward(self, s, src_key_padding_mask, need_weights=False):
        s = self._forward_latent(s, src_key_padding_mask, need_weights)
        # s: [E, T, N, H]

        if need_weights:
            value, attn_weight = self.value_layer(s)
            # value: [E, T, N, 1]
            return value, attn_weight

        value = self.value_layer(s)

        return value


# Using ff for temporal feature and positional encoding
class BaseTransformer_v3(nn.Module):
    def __init__(self, args):
        super(BaseTransformer_v3, self).__init__()

        state_dim = deepcopy(args.state_dim)
        state_dim.pop('encoding')

        state_dim = sum([math.prod(shape) for shape in state_dim.values()])

        # print('state_dim:', state_dim)
        # exit()

        # self.fc1 = nn.Linear(in_features=state_dim, out_features=args.lstm_hidden_dim)
        self.fc1 = nn.Sequential(
            nn.Linear(state_dim, args.lstm_hidden_dim),
            nn.ReLU(),
        )

        self.use_transformer = True

        self.is_decentralized = False

        if self.use_transformer:
            self.fc2 = nn.Sequential(
                nn.Linear(args.lstm_hidden_dim, args.transformer_hidden_dim),
                nn.ReLU(),
            )

            self.positional_encoder = nn.Linear(2, args.transformer_hidden_dim)

            self.encoder_layer = nn.TransformerEncoderLayer(d_model=args.transformer_hidden_dim,
                                                            nhead=args.transformer_num_heads,
                                                            dim_feedforward=args.transformer_dim_feedforward,
                                                            batch_first=True)

            self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=args.transformer_num_layers)

    def _forward_latent(self, s, src_key_padding_mask, need_weights=False):
        positional_encoding = s['encoding'][:, :, :, 0]
        # positional_encoding: [E, T, N, 2], selecting first encoding in the state history
        # as encoding is same throughout a window

        s = torch.cat([v for k, v in s.items() if k != 'encoding'], dim=-1)
        # s: [E, T, N, *F]

        E, T, N, *F = s.size()

        s = s.transpose(1, 2)
        # s: [E, N, T, *F]

        s = s.reshape(E * N, T, -1)
        # s: [E * N, T, prod(*F)]

        # print('sizes', E, T, N, F)
        # exit()

        s = self.fc1(s)
        # s: [E * N, T, H=hidden_size_lstm]

        if self.use_transformer:
            s = self.fc2(s)
            # s: [E * N, T, H=transformer_hidden_dim]

        s = s.view(E, N, T, -1)
        # s: [E, N, T, H]

        s = s.transpose(1, 2)
        # s: [E, T, N, H]

        if self.use_transformer:
            # s = s + positional_encoding
            positional_encoding = self.positional_encoder(positional_encoding)

            s = s + positional_encoding

            s = s.reshape(E * T, N, -1)
            # s: [E * T, N, H]

            src_key_padding_mask = src_key_padding_mask.view(E * T, N)
            # src_key_padding_mask: [E * T, N]

            active_seq_mask = (~src_key_padding_mask).any(-1)
            # active_seq_idx: [E * T]

            # Prevents in-place modification for computed gradients
            s_out = s.clone()

            if self.is_decentralized:
                mask = ~torch.eye(N, dtype=torch.bool, device=s.device)
            else:
                mask = torch.zeros(N, N, dtype=torch.bool, device=s.device)

            if need_weights:
                s_out[active_seq_mask], attn_weight = self.transformer_encoder(s[active_seq_mask],
                                                                               mask=mask,
                                                                               src_key_padding_mask=
                                                                               src_key_padding_mask[
                                                                                   active_seq_mask],
                                                                               need_weights=need_weights)
            else:
                s_out[active_seq_mask] = self.transformer_encoder(s[active_seq_mask],
                                                                  mask=mask,
                                                                  src_key_padding_mask=src_key_padding_mask[
                                                                      active_seq_mask])
            s = s_out
            # s: [E * T, N, H]

            s = s.view(E, T, N, -1)
            # s: [E, T, N, H]

        if need_weights:
            return s, attn_weight

        return s

    def forward(self, s, src_key_padding_mask, need_weights=False):
        raise NotImplementedError


class Actor_Transformer_v3(BaseTransformer_v3):
    def __init__(self, args):
        super(Actor_Transformer_v3, self).__init__(args)

        self.mean_layer = nn.Linear(args.transformer_hidden_dim, args.action_dim)

        if args.std:
            self.log_std = nn.Parameter(torch.tensor(args.std).log(), requires_grad=False)
        else:
            self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))

        if args.use_orthogonal_init:
            if hasattr(self, 'mean_layer'):
                orthogonal_init(self.mean_layer, gain=0.01)

            if hasattr(self, 'std_layer'):
                orthogonal_init(self.std_layer, gain=0.01)

    # s: [E=num_episodes/batch_size, T=seq_len, N=num_cassie, *F=*num_features]
    def forward(self, s, src_key_padding_mask, need_weights=False):
        if need_weights:
            s, attn_weight = self._forward_latent(s, src_key_padding_mask, need_weights)
        else:
            s = self._forward_latent(s, src_key_padding_mask, need_weights)
            # s: [E, T, N, H]

        mean = self.mean_layer(s)
        # mean: [E, T, N, action_dim]

        if need_weights:
            return mean, self.log_std.expand_as(mean).exp(), attn_weight

        return mean, self.log_std.expand_as(mean).exp()

    def pdf(self, s, src_key_padding_mask, need_weights=False):

        if need_weights:
            mean, std, attn_weight = self.forward(s, src_key_padding_mask, need_weights)
            return Normal(mean, std), attn_weight

        mean, std = self.forward(s, src_key_padding_mask, need_weights)
        return Normal(mean, std)


class Critic_Transformer_v3(BaseTransformer_v3):
    def __init__(self, args):
        super(Critic_Transformer_v3, self).__init__(args)

        self.value_layer = nn.Linear(args.transformer_hidden_dim, 1)

        if args.use_orthogonal_init:
            if hasattr(self, 'value_layer'):
                orthogonal_init(self.value_layer, gain=0.01)

    def forward(self, s, src_key_padding_mask, need_weights=False):
        s = self._forward_latent(s, src_key_padding_mask, need_weights)
        # s: [E, T, N, H]

        if need_weights:
            value, attn_weight = self.value_layer(s)
            # value: [E, T, N, 1]
            return value, attn_weight

        value = self.value_layer(s)

        return value


# Using conv for temporal feature and positional encoding
class BaseTransformer_v4(nn.Module):
    def __init__(self, args):
        super(BaseTransformer_v4, self).__init__()

        state_dim = deepcopy(args.state_dim)
        state_dim.pop('encoding')

        state_dim = sum([shape[-1] for shape in state_dim.values()])

        # model: conv1
        # self.conv1 = nn.Sequential(
        #     nn.Conv1d(in_channels=state_dim, out_channels=32, kernel_size=3, stride=1, padding=0),
        #     nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=0),
        #     nn.ReLU(),
        #     nn.Flatten(),
        #     nn.Linear(in_features=736, out_features=64),
        # )

        # model: conv2
        # self.conv1 = nn.Sequential(
        #     nn.Conv1d(in_channels=41, out_channels=4, kernel_size=3, stride=1, padding=1),
        #     nn.Conv1d(in_channels=4, out_channels=1, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Flatten(),
        #     nn.Linear(in_features=50, out_features=64),
        # )

        # model: conv3
        # self.conv1 = nn.Sequential(
        #     nn.Conv1d(in_channels=state_dim, out_channels=1, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Flatten(),
        #     nn.Linear(in_features=500, out_features=64),
        # )

        # model: conv4
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=state_dim, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.Flatten(),
            nn.Linear(in_features=800, out_features=64),
            nn.ReLU()
        )

        self.use_transformer = False

        self.is_decentralized = False

        if self.use_transformer:
            self.fc2 = nn.Sequential(
                nn.Linear(args.lstm_hidden_dim, args.transformer_hidden_dim),
                nn.ReLU(),
            )

            self.positional_encoder = nn.Linear(2, args.transformer_hidden_dim)

            self.encoder_layer = nn.TransformerEncoderLayer(d_model=args.transformer_hidden_dim,
                                                            nhead=args.transformer_num_heads,
                                                            dim_feedforward=args.transformer_dim_feedforward,
                                                            batch_first=True)

            self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=args.transformer_num_layers)

    def _forward_latent(self, s, src_key_padding_mask, need_weights=False):
        positional_encoding = s['encoding'][:, :, :, 0]
        # positional_encoding: [E, T, N, 2], selecting first encoding in the state history
        # as encoding is same throughout a window

        s = torch.cat([v for k, v in s.items() if k != 'encoding'], dim=-1)
        # s: [E, T, N, W, F]

        E, T, N, W, F = s.size()

        s = s.transpose(1, 2)
        # s: [E, N, T, W, F]

        s = s.reshape(E * N * T, W, F)
        # s: [E * N * T, W, F]

        s = s.transpose(-1, -2)
        # s: [E * N * T, F, W]

        s = self.conv1(s)
        # s: [E * N * T, H=hidden_size_lstm]

        if self.use_transformer:
            s = self.fc2(s)
            # s: [E * N * T, H=transformer_hidden_dim]

        s = s.view(E, N, T, -1)
        # s: [E, N, T, H]

        s = s.transpose(1, 2)
        # s: [E, T, N, H]

        if self.use_transformer:
            # s = s + positional_encoding
            positional_encoding = self.positional_encoder(positional_encoding)

            s = s + positional_encoding

            s = s.reshape(E * T, N, -1)
            # s: [E * T, N, H]

            src_key_padding_mask = src_key_padding_mask.view(E * T, N)
            # src_key_padding_mask: [E * T, N]

            active_seq_mask = (~src_key_padding_mask).any(-1)
            # active_seq_idx: [E * T]

            # Prevents in-place modification for computed gradients
            s_out = s.clone()

            if self.is_decentralized:
                mask = ~torch.eye(N, dtype=torch.bool, device=s.device)
            else:
                mask = torch.zeros(N, N, dtype=torch.bool, device=s.device)

            if need_weights:
                s_out[active_seq_mask], attn_weight = self.transformer_encoder(s[active_seq_mask],
                                                                               mask=mask,
                                                                               src_key_padding_mask=
                                                                               src_key_padding_mask[
                                                                                   active_seq_mask],
                                                                               need_weights=need_weights)
            else:
                s_out[active_seq_mask] = self.transformer_encoder(s[active_seq_mask],
                                                                  mask=mask,
                                                                  src_key_padding_mask=src_key_padding_mask[
                                                                      active_seq_mask])
            s = s_out
            # s: [E * T, N, H]

            s = s.view(E, T, N, -1)
            # s: [E, T, N, H]

        if need_weights:
            return s, attn_weight

        return s

    def forward(self, s, src_key_padding_mask, need_weights=False):
        raise NotImplementedError


class Actor_Transformer_v4(BaseTransformer_v4):
    def __init__(self, args):
        super(Actor_Transformer_v4, self).__init__(args)

        self.mean_layer = nn.Linear(args.transformer_hidden_dim, args.action_dim)

        if args.std:
            self.log_std = nn.Parameter(torch.tensor(args.std).log(), requires_grad=False)
        else:
            self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))

        if args.use_orthogonal_init:
            if hasattr(self, 'mean_layer'):
                orthogonal_init(self.mean_layer, gain=0.01)

            if hasattr(self, 'std_layer'):
                orthogonal_init(self.std_layer, gain=0.01)

    # s: [E=num_episodes/batch_size, T=seq_len, N=num_cassie, *F=*num_features]
    def forward(self, s, src_key_padding_mask, need_weights=False):
        if need_weights:
            s, attn_weight = self._forward_latent(s, src_key_padding_mask, need_weights)
        else:
            s = self._forward_latent(s, src_key_padding_mask, need_weights)
            # s: [E, T, N, H]

        mean = self.mean_layer(s)
        # mean: [E, T, N, action_dim]

        if need_weights:
            return mean, self.log_std.expand_as(mean).exp(), attn_weight

        return mean, self.log_std.expand_as(mean).exp()

    def pdf(self, s, src_key_padding_mask, need_weights=False):

        if need_weights:
            mean, std, attn_weight = self.forward(s, src_key_padding_mask, need_weights)
            return Normal(mean, std), attn_weight

        mean, std = self.forward(s, src_key_padding_mask, need_weights)
        return Normal(mean, std)


class Critic_Transformer_v4(BaseTransformer_v4):
    def __init__(self, args):
        super(Critic_Transformer_v4, self).__init__(args)

        self.value_layer = nn.Linear(args.transformer_hidden_dim, 1)

        if args.use_orthogonal_init:
            if hasattr(self, 'value_layer'):
                orthogonal_init(self.value_layer, gain=0.01)

    def forward(self, s, src_key_padding_mask, need_weights=False):
        s = self._forward_latent(s, src_key_padding_mask, need_weights)
        # s: [E, T, N, H]

        if need_weights:
            value, attn_weight = self.value_layer(s)
            # value: [E, T, N, 1]
            return value, attn_weight

        value = self.value_layer(s)

        return value


# Using ff for temporal feature and positional encoding
class BaseTransformer_v5(nn.Module):
    def __init__(self, args):
        super(BaseTransformer_v5, self).__init__()

        state_dim = deepcopy(args.state_dim)
        state_dim.pop('encoding')

        state_dim = sum([math.prod(shape) for shape in state_dim.values()])

        # self.fc1 = nn.Linear(in_features=state_dim, out_features=args.lstm_hidden_dim)
        self.fc1 = nn.Sequential(
            nn.Linear(state_dim, args.lstm_hidden_dim),
            nn.ReLU(),
        )

        self.use_transformer = True

        self.is_decentralized = False

        if self.use_transformer:
            self.fc2 = nn.Sequential(
                nn.Linear(args.lstm_hidden_dim, args.transformer_hidden_dim),
                nn.ReLU(),
            )

            self.encoder_layer = nn.TransformerEncoderLayer(d_model=args.transformer_hidden_dim + 2,
                                                            nhead=args.transformer_num_heads,
                                                            dim_feedforward=args.transformer_dim_feedforward,
                                                            batch_first=True)

            self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=args.transformer_num_layers)

    def _forward_latent(self, s, src_key_padding_mask, need_weights=False):
        positional_encoding = s['encoding'][:, :, :, 0]
        # positional_encoding: [E, T, N, 2], selecting first encoding in the state history
        # as encoding is same throughout a window

        s = torch.cat([v for k, v in s.items() if k != 'encoding'], dim=-1)
        # s: [E, T, N, *F]

        E, T, N, *F = s.size()

        s = s.transpose(1, 2)
        # s: [E, N, T, *F]

        s = s.reshape(E * N, T, -1)
        # s: [E * N, T, prod(*F)]

        s = self.fc1(s)
        # s: [E * N, T, H=hidden_size_lstm]

        if self.use_transformer:
            s = self.fc2(s)
            # s: [E * N, T, H=transformer_hidden_dim]

        s = s.view(E, N, T, -1)
        # s: [E, N, T, H]

        s = s.transpose(1, 2)
        # s: [E, T, N, H]

        if self.use_transformer:
            s = torch.cat((s, positional_encoding), dim=-1)

            s = s.reshape(E * T, N, -1)
            # s: [E * T, N, H+2]

            src_key_padding_mask = src_key_padding_mask.view(E * T, N)
            # src_key_padding_mask: [E * T, N]

            active_seq_mask = (~src_key_padding_mask).any(-1)
            # active_seq_idx: [E * T]

            # Prevents in-place modification for computed gradients
            s_out = s.clone()

            if self.is_decentralized:
                mask = ~torch.eye(N, dtype=torch.bool, device=s.device)
            else:
                mask = torch.zeros(N, N, dtype=torch.bool, device=s.device)

            if need_weights:
                s_out[active_seq_mask], attn_weight = self.transformer_encoder(s[active_seq_mask],
                                                                               mask=mask,
                                                                               src_key_padding_mask=
                                                                               src_key_padding_mask[
                                                                                   active_seq_mask],
                                                                               need_weights=need_weights)
            else:
                s_out[active_seq_mask] = self.transformer_encoder(s[active_seq_mask],
                                                                  mask=mask,
                                                                  src_key_padding_mask=src_key_padding_mask[
                                                                      active_seq_mask])
            s = s_out
            # s: [E * T, N, H]

            s = s.view(E, T, N, -1)
            # s: [E, T, N, H]

        if need_weights:
            return s, attn_weight

        return s

    def forward(self, s, src_key_padding_mask, need_weights=False):
        raise NotImplementedError


class Actor_Transformer_v5(BaseTransformer_v5):
    def __init__(self, args):
        super(Actor_Transformer_v5, self).__init__(args)

        self.mean_layer = nn.Linear(args.transformer_hidden_dim + 2, args.action_dim)

        if args.std:
            self.log_std = nn.Parameter(torch.tensor(args.std).log(), requires_grad=False)
        else:
            self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))

        if args.use_orthogonal_init:
            if hasattr(self, 'mean_layer'):
                orthogonal_init(self.mean_layer, gain=0.01)

            if hasattr(self, 'std_layer'):
                orthogonal_init(self.std_layer, gain=0.01)

    # s: [E=num_episodes/batch_size, T=seq_len, N=num_cassie, *F=*num_features]
    def forward(self, s, src_key_padding_mask, need_weights=False):
        if need_weights:
            s, attn_weight = self._forward_latent(s, src_key_padding_mask, need_weights)
        else:
            s = self._forward_latent(s, src_key_padding_mask, need_weights)
            # s: [E, T, N, H]

        mean = self.mean_layer(s)
        # mean: [E, T, N, action_dim]

        if need_weights:
            return mean, self.log_std.expand_as(mean).exp(), attn_weight

        return mean, self.log_std.expand_as(mean).exp()

    def pdf(self, s, src_key_padding_mask, need_weights=False):

        if need_weights:
            mean, std, attn_weight = self.forward(s, src_key_padding_mask, need_weights)
            return Normal(mean, std), attn_weight

        mean, std = self.forward(s, src_key_padding_mask, need_weights)
        return Normal(mean, std)


class Critic_Transformer_v5(BaseTransformer_v5):
    def __init__(self, args):
        super(Critic_Transformer_v5, self).__init__(args)

        self.value_layer = nn.Linear(args.transformer_hidden_dim + 2, 1)

        if args.use_orthogonal_init:
            if hasattr(self, 'value_layer'):
                orthogonal_init(self.value_layer, gain=0.01)

    def forward(self, s, src_key_padding_mask, need_weights=False):
        s = self._forward_latent(s, src_key_padding_mask, need_weights)
        # s: [E, T, N, H]

        if need_weights:
            value, attn_weight = self.value_layer(s)
            # value: [E, T, N, 1]
            return value, attn_weight

        value = self.value_layer(s)

        return value


# Using ff for temporal feature and positional encoding
class BaseTransformer_v6(nn.Module):
    def __init__(self, args):
        super(BaseTransformer_v6, self).__init__()

        state_dim = deepcopy(args.state_dim)
        state_dim.pop('encoding')

        state_dim = sum([math.prod(shape) for shape in state_dim.values()])

        # print('state_dim:', state_dim)
        # exit()

        # self.fc1 = nn.Linear(in_features=state_dim, out_features=args.lstm_hidden_dim)
        self.fc1 = nn.Sequential(
            nn.Linear(state_dim, args.lstm_hidden_dim),
            nn.ReLU(),
        )

        self.use_transformer = True

        self.is_decentralized = False

        if self.use_transformer:
            self.fc2 = nn.Sequential(
                nn.Linear(args.lstm_hidden_dim, args.transformer_hidden_dim),
                nn.ReLU(),
            )

            self.positional_encoder = nn.Linear(2, 16)

            self.encoder_layer = nn.TransformerEncoderLayer(d_model=args.transformer_hidden_dim + 16,
                                                            nhead=args.transformer_num_heads,
                                                            dim_feedforward=args.transformer_dim_feedforward,
                                                            batch_first=True)

            self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=args.transformer_num_layers)

    def _forward_latent(self, s, src_key_padding_mask, need_weights=False):
        positional_encoding = s['encoding'][:, :, :, 0]
        # positional_encoding: [E, T, N, 2], selecting first encoding in the state history
        # as encoding is same throughout a window

        s = torch.cat([v for k, v in s.items() if k != 'encoding'], dim=-1)
        # s: [E, T, N, *F]

        E, T, N, *F = s.size()

        s = s.transpose(1, 2)
        # s: [E, N, T, *F]

        s = s.reshape(E * N, T, -1)
        # s: [E * N, T, prod(*F)]

        # print('sizes', E, T, N, F)
        # exit()

        s = self.fc1(s)
        # s: [E * N, T, H=hidden_size_lstm]

        if self.use_transformer:
            s = self.fc2(s)
            # s: [E * N, T, H=transformer_hidden_dim]

        s = s.view(E, N, T, -1)
        # s: [E, N, T, H]

        s = s.transpose(1, 2)
        # s: [E, T, N, H]

        if self.use_transformer:
            # s = s + positional_encoding
            positional_encoding = self.positional_encoder(positional_encoding)

            s = torch.cat((s, positional_encoding), dim=-1)

            s = s.reshape(E * T, N, -1)
            # s: [E * T, N, H+16]

            src_key_padding_mask = src_key_padding_mask.view(E * T, N)
            # src_key_padding_mask: [E * T, N]

            active_seq_mask = (~src_key_padding_mask).any(-1)
            # active_seq_idx: [E * T]

            # Prevents in-place modification for computed gradients
            s_out = s.clone()

            if self.is_decentralized:
                mask = ~torch.eye(N, dtype=torch.bool, device=s.device)
            else:
                mask = torch.zeros(N, N, dtype=torch.bool, device=s.device)

            if need_weights:
                s_out[active_seq_mask], attn_weight = self.transformer_encoder(s[active_seq_mask],
                                                                               mask=mask,
                                                                               src_key_padding_mask=
                                                                               src_key_padding_mask[
                                                                                   active_seq_mask],
                                                                               need_weights=need_weights)
            else:
                s_out[active_seq_mask] = self.transformer_encoder(s[active_seq_mask],
                                                                  mask=mask,
                                                                  src_key_padding_mask=src_key_padding_mask[
                                                                      active_seq_mask])
            s = s_out
            # s: [E * T, N, H]

            s = s.view(E, T, N, -1)
            # s: [E, T, N, H]

        if need_weights:
            return s, attn_weight

        return s

    def forward(self, s, src_key_padding_mask, need_weights=False):
        raise NotImplementedError


class Actor_Transformer_v6(BaseTransformer_v6):
    def __init__(self, args):
        super(Actor_Transformer_v6, self).__init__(args)

        self.mean_layer = nn.Linear(args.transformer_hidden_dim + 16, args.action_dim)

        if args.std:
            self.log_std = nn.Parameter(torch.tensor(args.std).log(), requires_grad=False)
        else:
            self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))

        if args.use_orthogonal_init:
            if hasattr(self, 'mean_layer'):
                orthogonal_init(self.mean_layer, gain=0.01)

            if hasattr(self, 'std_layer'):
                orthogonal_init(self.std_layer, gain=0.01)

    # s: [E=num_episodes/batch_size, T=seq_len, N=num_cassie, *F=*num_features]
    def forward(self, s, src_key_padding_mask, need_weights=False):
        if need_weights:
            s, attn_weight = self._forward_latent(s, src_key_padding_mask, need_weights)
        else:
            s = self._forward_latent(s, src_key_padding_mask, need_weights)
            # s: [E, T, N, H+16]

        mean = self.mean_layer(s)
        # mean: [E, T, N, action_dim]

        if need_weights:
            return mean, self.log_std.expand_as(mean).exp(), attn_weight

        return mean, self.log_std.expand_as(mean).exp()

    def pdf(self, s, src_key_padding_mask, need_weights=False):

        if need_weights:
            mean, std, attn_weight = self.forward(s, src_key_padding_mask, need_weights)
            return Normal(mean, std), attn_weight

        mean, std = self.forward(s, src_key_padding_mask, need_weights)
        return Normal(mean, std)


class Critic_Transformer_v6(BaseTransformer_v6):
    def __init__(self, args):
        super(Critic_Transformer_v6, self).__init__(args)

        self.value_layer = nn.Linear(args.transformer_hidden_dim + 16, 1)

        if args.use_orthogonal_init:
            if hasattr(self, 'value_layer'):
                orthogonal_init(self.value_layer, gain=0.01)

    def forward(self, s, src_key_padding_mask, need_weights=False):
        s = self._forward_latent(s, src_key_padding_mask, need_weights)
        # s: [E, T, N, H+16]

        if need_weights:
            value, attn_weight = self.value_layer(s)
            # value: [E, T, N, 1]
            return value, attn_weight

        value = self.value_layer(s)

        return value


class BaseTransformer_v7(nn.Module):
    def __init__(self, args):
        super(BaseTransformer_v7, self).__init__()

        state_dim = deepcopy(args.state_dim)
        state_dim.pop('encoding')
        if 'hfield' in state_dim:
            del state_dim['hfield']

        state_dim = sum([math.prod(shape) for shape in state_dim.values()])

        self.lstm = nn.LSTM(input_size=state_dim, hidden_size=args.lstm_hidden_dim,
                            num_layers=args.lstm_num_layers, batch_first=True)

        self.use_transformer = True

        self.is_decentralized = False

        if self.use_transformer:
            self.fc1 = nn.Sequential(
                nn.Linear(args.lstm_hidden_dim, args.transformer_hidden_dim),
                nn.ReLU(),
            )

            self.positional_encoder = nn.Linear(2, 16)

            self.encoder_layer = nn.TransformerEncoderLayer(d_model=args.transformer_hidden_dim + 16,
                                                            nhead=args.transformer_num_heads,
                                                            batch_first=True)

            self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=args.transformer_num_layers)

    def init_hidden_state(self, device=torch.device('cpu'), batch_size=None):
        if batch_size is None:
            self.hidden_state = torch.zeros(self.lstm.num_layers, self.lstm.hidden_size).to(device)
            self.cell_state = torch.zeros(self.lstm.num_layers, self.lstm.hidden_size).to(device)
        else:
            self.hidden_state = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device)
            self.cell_state = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device)

    def _forward_latent(self, s, src_key_padding_mask, need_weights=False):
        positional_encoding = s['encoding'][:, :, :, 0]
        # positional_encoding: [E, T, N, 2], selecting first encoding in the state history
        # as encoding is same throughout a window

        s = torch.cat([v for k, v in s.items() if k not in ('encoding', 'hfield')], dim=-1)
        # s: [E, T, N, *F]

        E, T, N, *F = s.size()

        s = s.transpose(1, 2)
        # s: [E, N, T, *F]

        s = s.reshape(E * N, T, -1)
        # s: [E * N, T, prod(*F)]

        s, (self.hidden_state, self.cell_state) = self.lstm(s, (self.hidden_state, self.cell_state))
        # s: [E * N, T, H=hidden_size_lstm]

        if self.use_transformer:
            s = self.fc1(s)
            # s: [E * N, T, H=transformer_hidden_dim]

        s = s.view(E, N, T, -1)
        # s: [E, N, T, H]

        s = s.transpose(1, 2)
        # s: [E, T, N, H]

        if self.use_transformer:
            # s = s + positional_encoding
            positional_encoding = self.positional_encoder(positional_encoding)

            s = torch.cat((s, positional_encoding), dim=-1)

            s = s.reshape(E * T, N, -1)
            # s: [E * T, N, H+16]

            src_key_padding_mask = src_key_padding_mask.view(E * T, N)
            # src_key_padding_mask: [E * T, N]

            active_seq_mask = (~src_key_padding_mask).any(-1)
            # active_seq_idx: [E * T]

            # Prevents in-place modification for computed gradients
            s_out = s.clone()

            if self.is_decentralized:
                mask = ~torch.eye(N, dtype=torch.bool, device=s.device)
            else:
                mask = torch.zeros(N, N, dtype=torch.bool, device=s.device)

            if need_weights:
                s_out[active_seq_mask], attn_weight = self.transformer_encoder(s[active_seq_mask],
                                                                               mask=mask,
                                                                               src_key_padding_mask=
                                                                               src_key_padding_mask[
                                                                                   active_seq_mask],
                                                                               need_weights=need_weights)
            else:
                s_out[active_seq_mask] = self.transformer_encoder(s[active_seq_mask],
                                                                  mask=mask,
                                                                  src_key_padding_mask=src_key_padding_mask[
                                                                      active_seq_mask])
            s = s_out
            # s: [E * T, N, H]

            s = s.view(E, T, N, -1)
            # s: [E, T, N, H]

        if need_weights:
            return s, attn_weight

        return s

    def forward(self, s, src_key_padding_mask, need_weights=False):
        raise NotImplementedError


class Actor_Transformer_v7(BaseTransformer_v7):
    def __init__(self, args):
        super(Actor_Transformer_v7, self).__init__(args)

        self.mean_layer = nn.Linear(args.transformer_hidden_dim + 16, args.action_dim)

        if args.std:
            self.log_std = nn.Parameter(torch.tensor(args.std).log(), requires_grad=False)
        else:
            self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))

        if args.use_orthogonal_init:
            if hasattr(self, 'mean_layer'):
                orthogonal_init(self.mean_layer, gain=0.01)

            if hasattr(self, 'std_layer'):
                orthogonal_init(self.std_layer, gain=0.01)

    # s: [E=num_episodes/batch_size, T=seq_len, N=num_cassie, *F=*num_features]
    def forward(self, s, src_key_padding_mask, need_weights=False):
        if need_weights:
            s, attn_weight = self._forward_latent(s, src_key_padding_mask, need_weights)
        else:
            s = self._forward_latent(s, src_key_padding_mask, need_weights)
            # s: [E, T, N, H]

        mean = self.mean_layer(s)
        # mean: [E, T, N, action_dim]

        if need_weights:
            return mean, self.log_std.expand_as(mean).exp(), attn_weight

        return mean, self.log_std.expand_as(mean).exp()

    def pdf(self, s, src_key_padding_mask, need_weights=False):

        if need_weights:
            mean, std, attn_weight = self.forward(s, src_key_padding_mask, need_weights)
            return Normal(mean, std), attn_weight

        mean, std = self.forward(s, src_key_padding_mask, need_weights)

        return Normal(mean, std)


class Critic_Transformer_v7(BaseTransformer_v7):
    def __init__(self, args):
        super(Critic_Transformer_v7, self).__init__(args)

        self.value_layer = nn.Linear(args.transformer_hidden_dim + 16, 1)

        if args.use_orthogonal_init:
            if hasattr(self, 'value_layer'):
                orthogonal_init(self.value_layer, gain=0.01)

    def forward(self, s, src_key_padding_mask, need_weights=False):
        s = self._forward_latent(s, src_key_padding_mask, need_weights)
        # s: [E, T, N, H]

        if need_weights:
            value, attn_weight = self.value_layer(s)
            # value: [E, T, N, 1]
            return value, attn_weight

        value = self.value_layer(s)

        return value


class BaseTransformer_v8(nn.Module):
    class TransformerBlock(nn.Module):
        def __init__(self, args):
            super(BaseTransformer_v8.TransformerBlock, self).__init__()

            self.fc1 = nn.Sequential(
                nn.Linear(args.lstm_hidden_dim, args.transformer_hidden_dim),
                nn.ReLU(),
            )

            self.positional_encoder = nn.Linear(2, 16)

            self.encoder_layer = nn.TransformerEncoderLayer(d_model=args.transformer_hidden_dim + 16,
                                                            nhead=args.transformer_num_heads,
                                                            batch_first=True)

            self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=args.transformer_num_layers)

            self.compressor = nn.Linear(args.transformer_hidden_dim + 16, args.transformer_hidden_dim)

        def forward(self, s, positional_encoding, src_key_padding_mask, need_weights=False):
            # s: [E, T, N, H=latent_dim]
            # positional_encoding: [E, T, N, 2]
            # src_key_padding_mask: [E, T, N]

            E, T, N = s.size()[:3]

            s = self.fc1(s)
            # s: [E, T, N, H=transformer_hidden_dim]

            positional_encoding = self.positional_encoder(positional_encoding)
            # positional_encoding: [E, T, N, 16]

            s = torch.cat((s, positional_encoding), dim=-1)
            # s: [E, T, N, H+16]

            s = s.reshape(E * T, N, -1)
            # s: [E * T, N, H+16]

            src_key_padding_mask = src_key_padding_mask.view(E * T, N)
            # src_key_padding_mask: [E * T, N]

            active_seq_mask = (~src_key_padding_mask).any(-1)
            # active_seq_idx: [E * T]

            # Prevents in-place modification for computed gradients
            s_out = s.clone()

            if need_weights:
                s_out[active_seq_mask], attn_weight = self.transformer_encoder(s[active_seq_mask],
                                                                               src_key_padding_mask=
                                                                               src_key_padding_mask[
                                                                                   active_seq_mask],
                                                                               need_weights=need_weights)
            else:
                s_out[active_seq_mask] = self.transformer_encoder(s[active_seq_mask],
                                                                  src_key_padding_mask=src_key_padding_mask[
                                                                      active_seq_mask])
            s = s_out
            # s: [E * T, N, H+16]

            s = s.view(E, T, N, -1)
            # s: [E, T, N, H+16]

            s = self.compressor(s)
            # s: [E, T, N, H]

            if need_weights:
                return s, attn_weight

            return s

    def __init__(self, args):
        super(BaseTransformer_v8, self).__init__()

        state_dim = deepcopy(args.state_dim)
        state_dim.pop('encoding')

        state_dim = sum([math.prod(shape) for shape in state_dim.values()])

        self.lstm = nn.LSTM(input_size=state_dim, hidden_size=args.lstm_hidden_dim,
                            num_layers=args.lstm_num_layers, batch_first=True)

        self.learn_lstm = True
        self.learn_transformer = False

        assert self.learn_lstm or self.learn_transformer, 'At least one of LSTM or Transformer should be used'

        if not self.learn_lstm:
            for param in self.lstm.parameters():
                param.requires_grad = False

        self.transformer_block = self.TransformerBlock(args)

    def init_hidden_state(self, device=torch.device('cpu'), batch_size=None):
        if batch_size is None:
            self.hidden_state = torch.zeros(self.lstm.num_layers, self.lstm.hidden_size).to(device)
            self.cell_state = torch.zeros(self.lstm.num_layers, self.lstm.hidden_size).to(device)
        else:
            self.hidden_state = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device)
            self.cell_state = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device)

    def _forward_latent(self, s, src_key_padding_mask, need_weights=False):
        positional_encoding = s['encoding'][:, :, :, 0]
        # positional_encoding: [E, T, N, 2], selecting first encoding in the state history
        # as encoding is same throughout a window

        s = torch.cat([v for k, v in s.items() if k != 'encoding'], dim=-1)
        # s: [E, T, N, *F]

        E, T, N, *F = s.size()

        s = s.transpose(1, 2)
        # s: [E, N, T, *F]

        s = s.reshape(E * N, T, -1)
        # s: [E * N, T, prod(*F)]

        s, (self.hidden_state, self.cell_state) = self.lstm(s, (self.hidden_state, self.cell_state))
        # s: [E * N, T, H=hidden_size_lstm]

        s = s.view(E, N, T, -1)
        # s: [E, N, T, H]

        s = s.transpose(1, 2)
        # s: [E, T, N, H]

        if self.learn_transformer:
            if need_weights:
                s, attn_weight = self.transformer_block(s, positional_encoding, src_key_padding_mask, need_weights)
                # s: [E, T, N, H=transformer_hidden_dim]
                # attn_weight: [E, T, N, N]

                return s, attn_weight

            s = self.transformer_block(s, positional_encoding, src_key_padding_mask)
            # s: [E, T, N, H=transformer_hidden_dim]

            return s

        if need_weights:
            return s, None

        return s

    def forward(self, s, src_key_padding_mask, need_weights=False):
        raise NotImplementedError


class Actor_Transformer_v8(BaseTransformer_v8):
    def __init__(self, args):
        super(Actor_Transformer_v8, self).__init__(args)

        self.mean_layer = nn.Linear(args.transformer_hidden_dim, args.action_dim)

        if args.std:
            self.log_std = nn.Parameter(torch.tensor(args.std).log(), requires_grad=False)
        else:
            self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))

        if args.use_orthogonal_init:
            if hasattr(self, 'mean_layer'):
                orthogonal_init(self.mean_layer, gain=0.01)

            if hasattr(self, 'std_layer'):
                orthogonal_init(self.std_layer, gain=0.01)

    # s: [E=num_episodes/batch_size, T=seq_len, N=num_cassie, *F=*num_features]
    def forward(self, s, src_key_padding_mask, need_weights=False):
        if need_weights:
            s, attn_weight = self._forward_latent(s, src_key_padding_mask, need_weights)
        else:
            s = self._forward_latent(s, src_key_padding_mask, need_weights)
            # s: [E, T, N, H]

        mean = self.mean_layer(s)
        # mean: [E, T, N, action_dim]

        if need_weights:
            return mean, self.log_std.expand_as(mean).exp(), attn_weight

        return mean, self.log_std.expand_as(mean).exp()

    def pdf(self, s, src_key_padding_mask, need_weights=False):

        if need_weights:
            mean, std, attn_weight = self.forward(s, src_key_padding_mask, need_weights)
            return Normal(mean, std), attn_weight

        mean, std = self.forward(s, src_key_padding_mask, need_weights)
        return Normal(mean, std)


class Critic_Transformer_v8(BaseTransformer_v8):
    def __init__(self, args):
        super(Critic_Transformer_v8, self).__init__(args)

        self.value_layer = nn.Linear(args.transformer_hidden_dim, 1)

        if args.use_orthogonal_init:
            if hasattr(self, 'value_layer'):
                orthogonal_init(self.value_layer, gain=0.01)

    def forward(self, s, src_key_padding_mask, need_weights=False):
        s = self._forward_latent(s, src_key_padding_mask, need_weights)
        # s: [E, T, N, H]

        if need_weights:
            value, attn_weight = self.value_layer(s)
            # value: [E, T, N, 1]
            return value, attn_weight

        value = self.value_layer(s)

        return value


class BaseTransformer_v9(nn.Module):
    class TransformerBlock(nn.Module):
        def __init__(self, args):
            super(BaseTransformer_v9.TransformerBlock, self).__init__()

            self.positional_encoder = nn.Linear(2, 16)
            self.command_encoder = nn.Linear(4, 16)

            self.encoder_layer = nn.TransformerEncoderLayer(d_model=args.transformer_hidden_dim,
                                                            nhead=args.transformer_num_heads,
                                                            batch_first=True)

            self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=args.transformer_num_layers)

        def forward(self, s, positional_encoding, cmd, src_key_padding_mask, need_weights=False):
            # s: [E, T, N, H=latent_dim]
            # positional_encoding: [E, T, N, 2]
            # positional_encoding: [E, T, 2]
            # src_key_padding_mask: [E, T, N]

            E, T, N = s.size()[:3]

            positional_encoding = self.positional_encoder(positional_encoding)
            # positional_encoding: [E, T, N, 16]

            cmd = self.command_encoder(cmd)
            # cmd: [E, T, N, 16]

            s = torch.cat((s, positional_encoding, cmd), dim=-1)
            # s: [E, T, N, H=transformer_hidden_dim]

            s = s.reshape(E * T, N, -1)
            # s: [E * T, N, H+16+16]

            src_key_padding_mask = src_key_padding_mask.view(E * T, N)
            # src_key_padding_mask: [E * T, N]

            active_seq_mask = (~src_key_padding_mask).any(-1)
            # active_seq_idx: [E * T]

            # Prevents in-place modification for computed gradients
            s_out = s.clone()

            if need_weights:
                s_out[active_seq_mask], attn_weight = self.transformer_encoder(s[active_seq_mask],
                                                                               src_key_padding_mask=
                                                                               src_key_padding_mask[
                                                                                   active_seq_mask],
                                                                               need_weights=need_weights)
            else:
                s_out[active_seq_mask] = self.transformer_encoder(s[active_seq_mask],
                                                                  src_key_padding_mask=src_key_padding_mask[
                                                                      active_seq_mask])
            s = s_out
            # s: [E * T, N, H+16]

            s = s.view(E, T, N, -1)
            # s: [E, T, N, H+16]

            if need_weights:
                return s, attn_weight

            return s

    def __init__(self, args):
        super(BaseTransformer_v9, self).__init__()

        state_dim = deepcopy(args.state_dim)
        state_dim.pop('encoding')
        state_dim.pop('cmd')
        if 'hfield' in state_dim:
            del state_dim['hfield']

        state_dim = sum([math.prod(shape) for shape in state_dim.values()])

        self.lstm = nn.LSTM(input_size=state_dim, hidden_size=args.lstm_hidden_dim,
                            num_layers=args.lstm_num_layers, batch_first=True)

        self.learn_lstm = True
        self.learn_transformer = True

        assert self.learn_lstm or self.learn_transformer, 'At least one of LSTM or Transformer should be used'

        if not self.learn_lstm:
            for param in self.lstm.parameters():
                param.requires_grad = False

        self.transformer_block = self.TransformerBlock(args)

    def init_hidden_state(self, device=torch.device('cpu'), batch_size=None):
        if batch_size is None:
            self.hidden_state = torch.zeros(self.lstm.num_layers, self.lstm.hidden_size).to(device)
            self.cell_state = torch.zeros(self.lstm.num_layers, self.lstm.hidden_size).to(device)
        else:
            self.hidden_state = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device)
            self.cell_state = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device)

    def _forward_latent(self, s, src_key_padding_mask, need_weights=False):
        positional_encoding = s['encoding'][:, :, :, 0]
        # positional_encoding: [E, T, N, 2], selecting first encoding in the state history
        # as encoding is same throughout a window
        cmd = s['cmd'][:, :, :, 0]
        # cmd: [E, T, N, 3]

        s = torch.cat([v for k, v in s.items() if k not in ('encoding', 'cmd', 'hfield')], dim=-1)
        # s: [E, T, N, W, F]

        E, T, N, *F = s.size()

        s = s.transpose(1, 2)
        # s: [E, N, T, W, F]

        s = s.reshape(E * N, T, -1)
        # s: [E * N, T, W * F]

        s, (self.hidden_state, self.cell_state) = self.lstm(s, (self.hidden_state, self.cell_state))
        # s: [E * N, T, H=hidden_size_lstm]

        s = s.view(E, N, T, -1)
        # s: [E, N, T, H]

        s = s.transpose(1, 2)
        # s: [E, T, N, H]

        if self.learn_transformer:
            if need_weights:
                s, attn_weight = self.transformer_block(s, positional_encoding, cmd, src_key_padding_mask, need_weights)
                # s: [E, T, N, H=transformer_hidden_dim]
                # attn_weight: [E, T, N, N]

                return s, attn_weight

            s = self.transformer_block(s, positional_encoding, cmd, src_key_padding_mask)
            # s: [E, T, N, H=transformer_hidden_dim]

            return s

        if need_weights:
            return s, None

        return s

    def forward(self, s, src_key_padding_mask, need_weights=False):
        raise NotImplementedError


class Actor_Transformer_v9(BaseTransformer_v9):
    def __init__(self, args):
        super(Actor_Transformer_v9, self).__init__(args)

        self.mean_layer = nn.Linear(args.transformer_hidden_dim, args.action_dim)

        if args.std:
            self.log_std = nn.Parameter(torch.tensor(args.std).log(), requires_grad=False)
        else:
            self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))

        if args.use_orthogonal_init:
            if hasattr(self, 'mean_layer'):
                orthogonal_init(self.mean_layer, gain=0.01)

            if hasattr(self, 'std_layer'):
                orthogonal_init(self.std_layer, gain=0.01)

    # s: [E=num_episodes/batch_size, T=seq_len, N=num_cassie, *F=*num_features]
    def forward(self, s, src_key_padding_mask, need_weights=False):
        if need_weights:
            s, attn_weight = self._forward_latent(s, src_key_padding_mask, need_weights)
        else:
            s = self._forward_latent(s, src_key_padding_mask, need_weights)
            # s: [E, T, N, H]

        mean = self.mean_layer(s)
        # mean: [E, T, N, action_dim]

        if need_weights:
            return mean, self.log_std.expand_as(mean).exp(), attn_weight

        return mean, self.log_std.expand_as(mean).exp()

    def pdf(self, s, src_key_padding_mask, need_weights=False):

        if need_weights:
            mean, std, attn_weight = self.forward(s, src_key_padding_mask, need_weights)
            return Normal(mean, std), attn_weight

        mean, std = self.forward(s, src_key_padding_mask, need_weights)
        return Normal(mean, std)


class Critic_Transformer_v9(BaseTransformer_v9):
    def __init__(self, args):
        super(Critic_Transformer_v9, self).__init__(args)

        self.value_layer = nn.Linear(args.transformer_hidden_dim, 1)

        if args.use_orthogonal_init:
            if hasattr(self, 'value_layer'):
                orthogonal_init(self.value_layer, gain=0.01)

    def forward(self, s, src_key_padding_mask, need_weights=False):
        s = self._forward_latent(s, src_key_padding_mask, need_weights)
        # s: [E, T, N, H]

        if need_weights:
            value, attn_weight = self.value_layer(s)
            # value: [E, T, N, 1]
            return value, attn_weight

        value = self.value_layer(s)

        return value


class BaseLatentLSTM(BaseLSTM):
    def __init__(self, args):
        super(BaseLatentLSTM, self).__init__()

        state_dim = sum([math.prod(shape) for shape in args.state_dim.values()])

        self.lstm = nn.LSTM(input_size=state_dim, hidden_size=args.lstm_hidden_dim,
                            num_layers=args.lstm_num_layers, batch_first=True)

    def _forward_latent(self, s):
        s = torch.cat(list(s.values()), dim=-1)
        # s: [E, T, N, *F]

        E, T, N, *F = s.size()

        s = s.transpose(1, 2)
        # s: [E, N, T, *F]

        s = s.reshape(E * N, T, -1)
        # s: [E * N, T, prod(*F)]

        s, (self.hidden_state, self.cell_state) = self.lstm(s, (self.hidden_state, self.cell_state))
        # s: [E * N, T, H]

        s = s.view(E, N, T, -1)
        # s: [E, N, T, H]

        s = s.transpose(1, 2)
        # s: [E, T, N, H]

        return s


class Actor_LSTM_v10(BaseLatentLSTM):
    def __init__(self, args):
        super(Actor_LSTM_v10, self).__init__(args)

        self.mean_layer = nn.Linear(args.lstm_hidden_dim, args.action_dim)

        if args.std:
            self.log_std = nn.Parameter(torch.tensor(args.std).log(), requires_grad=False)
        else:
            self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))

        if args.use_orthogonal_init:
            logging.info("------use_orthogonal_init------")
            if hasattr(self, 'mean_layer'):
                orthogonal_init(self.mean_layer, gain=0.01)

            if hasattr(self, 'std_layer'):
                orthogonal_init(self.std_layer, gain=0.01)

    def forward(self, s):
        s = self._forward_latent(s)
        # s: [E, T, N, H]

        mean = self.mean_layer(s)
        # mean: [E, T, N, action_dim]

        return mean, self.log_std.expand_as(mean).exp()

    def pdf(self, s):
        mean, std = self.forward(s)
        return Normal(mean, std)


class Critic_LSTM_v10(BaseLatentLSTM):
    def __init__(self, args):
        super(Critic_LSTM_v10, self).__init__(args)

        self.value_layer = nn.Linear(args.lstm_hidden_dim, 1)

        if args.use_orthogonal_init:
            logging.info("------use_orthogonal_init------")
            if hasattr(self, 'value_layer'):
                orthogonal_init(self.value_layer, gain=0.01)

    def forward(self, s):
        s = self._forward_latent(s)
        # s: [E, T, N, H]

        value = self.value_layer(s)
        # mean: [E, T, N, 1]

        return value


class BaseLSTM_v11(nn.Module):
    def __init__(self, args):
        super(BaseLSTM_v11, self).__init__()

        state_dim = deepcopy(args.state_dim)
        hfield_dim = state_dim.pop('hfield')

        state_dim = sum([math.prod(shape) for shape in state_dim.values()])

        self.lstm = nn.LSTM(input_size=state_dim, hidden_size=args.lstm_hidden_dim,
                            num_layers=args.lstm_num_layers, batch_first=True)

        self.lstm2 = nn.LSTM(input_size=math.prod(hfield_dim), hidden_size=args.lstm_hidden_dim,
                             num_layers=args.lstm_num_layers, batch_first=True)

        for param in self.lstm.parameters():
            param.requires_grad = False

    def init_hidden_state(self, device=torch.device('cpu'), batch_size=None):
        self.hidden_state, self.cell_state = self._init_hidden_state(self.lstm, device, batch_size)
        self.hidden_state2, self.cell_state2 = self._init_hidden_state(self.lstm2, device, batch_size)

    def _init_hidden_state(self, lstm, device=torch.device('cpu'), batch_size=None):
        if batch_size is None:
            hidden_state = torch.zeros(lstm.num_layers, lstm.hidden_size).to(device)
            cell_state = torch.zeros(lstm.num_layers, lstm.hidden_size).to(device)
        else:
            hidden_state = torch.zeros(lstm.num_layers, batch_size, lstm.hidden_size).to(device)
            cell_state = torch.zeros(lstm.num_layers, batch_size, lstm.hidden_size).to(device)

        return hidden_state, cell_state

    def _forward_hfield(self, s):
        # hfield shape: [E, T, N, W, 600]

        E, T, N, *F = s.size()

        s = s.transpose(1, 2)
        # s: [E, N, T, *F]

        s = s.reshape(E * N, T, -1)
        # s: [E * N, T, prod(*F)]

        s, (self.hidden_state2, self.cell_state2) = self.lstm2(s, (self.hidden_state2, self.cell_state2))
        # s: [E * N, T, H]

        s = s.view(E, N, T, -1)
        # s: [E, N, T, H]

        s = s.transpose(1, 2)
        # s: [E, T, N, H]

        return s

    def _forward_latent(self, s):
        hfield = s['hfield']
        # hfield: [E, T, N, 600]

        s = torch.cat([v for k, v in s.items() if k != 'hfield'], dim=-1)
        # s: [E, T, N, *F]

        E, T, N, *F = s.size()

        s = s.transpose(1, 2)
        # s: [E, N, T, *F]

        s = s.reshape(E * N, T, -1)
        # s: [E * N, T, prod(*F)]

        s, (self.hidden_state, self.cell_state) = self.lstm(s, (self.hidden_state, self.cell_state))
        # s: [E * N, T, H]

        s = s.view(E, N, T, -1)
        # s: [E, N, T, H]

        s = s.transpose(1, 2)
        # s: [E, T, N, H]

        s = s + self._forward_hfield(hfield)

        return s


class Actor_LSTM_v11(BaseLSTM_v11):
    def __init__(self, args):
        super(Actor_LSTM_v11, self).__init__(args)

        self.mean_layer = nn.Linear(args.lstm_hidden_dim, args.action_dim)

        if args.std:
            self.log_std = nn.Parameter(torch.tensor(args.std).log(), requires_grad=False)
        else:
            self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))

        if args.use_orthogonal_init:
            logging.info("------use_orthogonal_init------")
            if hasattr(self, 'mean_layer'):
                orthogonal_init(self.mean_layer, gain=0.01)

            if hasattr(self, 'std_layer'):
                orthogonal_init(self.std_layer, gain=0.01)

    def forward(self, s):
        s = self._forward_latent(s)
        # s: [E, T, N, H]

        mean = self.mean_layer(s)
        # mean: [E, T, N, action_dim]

        return mean, self.log_std.expand_as(mean).exp()

    def pdf(self, s):
        mean, std = self.forward(s)
        return Normal(mean, std)


class Critic_LSTM_v11(BaseLSTM_v11):
    def __init__(self, args):
        super(Critic_LSTM_v11, self).__init__(args)

        self.value_layer = nn.Sequential(nn.Linear(args.lstm_hidden_dim, 1))

        if args.use_orthogonal_init:
            logging.info("------use_orthogonal_init------")
            if hasattr(self, 'value_layer'):
                for layer in self.value_layer:
                    if isinstance(layer, nn.Linear):
                        orthogonal_init(layer, gain=0.01)

    def forward(self, s):
        s = self._forward_latent(s)
        # s: [E, T, N, H]

        value = self.value_layer(s)
        # mean: [E, T, N, 1]

        return value


class BaseLSTM_v12(nn.Module):
    def __init__(self, args):
        super(BaseLSTM_v12, self).__init__()

        state_dim = deepcopy(args.state_dim)
        hfield_dim = state_dim.pop('hfield')

        state_dim = sum([math.prod(shape) for shape in state_dim.values()])

        self.lstm = nn.LSTM(input_size=state_dim, hidden_size=args.lstm_hidden_dim,
                            num_layers=args.lstm_num_layers, batch_first=True)

        self.lstm2 = nn.LSTM(input_size=math.prod(hfield_dim), hidden_size=args.lstm_hidden_dim,
                             num_layers=args.lstm_num_layers, batch_first=True)

        # for param in self.lstm.parameters():
        #     param.requires_grad = False

    def init_hidden_state(self, device=torch.device('cpu'), batch_size=None):
        self.hidden_state, self.cell_state = self._init_hidden_state(self.lstm, device, batch_size)
        self.hidden_state2, self.cell_state2 = self._init_hidden_state(self.lstm2, device, batch_size)

    def _init_hidden_state(self, lstm, device=torch.device('cpu'), batch_size=None):
        if batch_size is None:
            hidden_state = torch.zeros(lstm.num_layers, lstm.hidden_size).to(device)
            cell_state = torch.zeros(lstm.num_layers, lstm.hidden_size).to(device)
        else:
            hidden_state = torch.zeros(lstm.num_layers, batch_size, lstm.hidden_size).to(device)
            cell_state = torch.zeros(lstm.num_layers, batch_size, lstm.hidden_size).to(device)

        return hidden_state, cell_state

    def _forward_hfield(self, s):
        # hfield shape: [E, T, N, W, 600]

        E, T, N, *F = s.size()

        s = s.transpose(1, 2)
        # s: [E, N, T, *F]

        s = s.reshape(E * N, T, -1)
        # s: [E * N, T, prod(*F)]

        s, (self.hidden_state2, self.cell_state2) = self.lstm2(s, (self.hidden_state2, self.cell_state2))
        # s: [E * N, T, H]

        s = s.view(E, N, T, -1)
        # s: [E, N, T, H]

        s = s.transpose(1, 2)
        # s: [E, T, N, H]

        return s

    def _forward_latent(self, s):
        hfield = s['hfield']
        # hfield: [E, T, N, 600]

        s = torch.cat([v for k, v in s.items() if k != 'hfield'], dim=-1)
        # s: [E, T, N, *F]

        E, T, N, *F = s.size()

        s = s.transpose(1, 2)
        # s: [E, N, T, *F]

        s = s.reshape(E * N, T, -1)
        # s: [E * N, T, prod(*F)]

        s, (self.hidden_state, self.cell_state) = self.lstm(s, (self.hidden_state, self.cell_state))
        # s: [E * N, T, H]

        s = s.view(E, N, T, -1)
        # s: [E, N, T, H]

        s = s.transpose(1, 2)
        # s: [E, T, N, H]

        return s, self._forward_hfield(hfield)


class Actor_LSTM_v12(BaseLSTM_v12):
    def __init__(self, args):
        super(Actor_LSTM_v12, self).__init__(args)

        self.mean_layer = nn.Linear(args.lstm_hidden_dim, args.action_dim)
        self.mean_layer2 = nn.Linear(args.lstm_hidden_dim, args.action_dim)

        if args.std:
            self.log_std = nn.Parameter(torch.tensor(args.std).log(), requires_grad=False)
            self.log_std2 = nn.Parameter(torch.tensor(args.std).log(), requires_grad=False)
        else:
            self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))
            self.log_std2 = nn.Parameter(torch.zeros(1, args.action_dim))

        if args.use_orthogonal_init:
            logging.info("------use_orthogonal_init------")
            if hasattr(self, 'mean_layer2'):
                orthogonal_init(self.mean_layer, gain=0.01)
                orthogonal_init(self.mean_layer2, gain=0.01)

            if hasattr(self, 'std_layer2'):
                orthogonal_init(self.std_layer, gain=0.01)
                orthogonal_init(self.std_layer2, gain=0.01)
        #
        # for param in self.mean_layer.parameters():
        #     param.requires_grad = False

    def forward(self, s):
        s, hfield = self._forward_latent(s)
        # s: [E, T, N, H]

        mean = self.mean_layer(s) + self.mean_layer2(hfield)
        # mean: [E, T, N, action_dim]

        return mean, self.log_std.expand_as(mean).exp()

    def pdf(self, s):
        mean, std = self.forward(s)
        return Normal(mean, std)


class Critic_LSTM_v12(BaseLSTM_v12):
    def __init__(self, args):
        super(Critic_LSTM_v12, self).__init__(args)

        self.value_layer = nn.Sequential(nn.Linear(args.lstm_hidden_dim, 1))

        self.value_layer2 = nn.Sequential(nn.Linear(args.lstm_hidden_dim, 1))

        if args.use_orthogonal_init:
            logging.info("------use_orthogonal_init------")
            if hasattr(self, 'value_layer'):
                for layer in self.value_layer:
                    if isinstance(layer, nn.Linear):
                        orthogonal_init(layer, gain=0.01)

        # for param in self.value_layer.parameters():
        #     param.requires_grad = False

    def forward(self, s):
        s, hfield = self._forward_latent(s)
        # s: [E, T, N, H]

        value = self.value_layer(s) + self.value_layer2(hfield)
        # mean: [E, T, N, 1]

        return value


class BaseLSTM_v13(nn.Module):
    def __init__(self, args):
        super(BaseLSTM_v13, self).__init__()

        state_dim = deepcopy(args.state_dim)
        hfield_dim = state_dim.pop('hfield')

        state_dim = sum([math.prod(shape) for shape in state_dim.values()])

        self.lstm = nn.LSTM(input_size=state_dim, hidden_size=args.lstm_hidden_dim,
                            num_layers=args.lstm_num_layers, batch_first=True)

        self.lstm2 = nn.LSTM(input_size=math.prod(hfield_dim), hidden_size=args.lstm_hidden_dim,
                             num_layers=args.lstm_num_layers, batch_first=True)

        # for param in self.lstm.parameters():
        #     param.requires_grad = False

    def init_hidden_state(self, device=torch.device('cpu'), batch_size=None):
        self.hidden_state, self.cell_state = self._init_hidden_state(self.lstm, device, batch_size)
        self.hidden_state2, self.cell_state2 = self._init_hidden_state(self.lstm2, device, batch_size)

    def _init_hidden_state(self, lstm, device=torch.device('cpu'), batch_size=None):
        if batch_size is None:
            hidden_state = torch.zeros(lstm.num_layers, lstm.hidden_size).to(device)
            cell_state = torch.zeros(lstm.num_layers, lstm.hidden_size).to(device)
        else:
            hidden_state = torch.zeros(lstm.num_layers, batch_size, lstm.hidden_size).to(device)
            cell_state = torch.zeros(lstm.num_layers, batch_size, lstm.hidden_size).to(device)

        return hidden_state, cell_state

    def _forward_hfield(self, s):
        # hfield shape: [E, T, N, W, 600]

        E, T, N, *F = s.size()

        s = s.transpose(1, 2)
        # s: [E, N, T, *F]

        s = s.reshape(E * N, T, -1)
        # s: [E * N, T, prod(*F)]

        s, (self.hidden_state2, self.cell_state2) = self.lstm2(s, (self.hidden_state2, self.cell_state2))
        # s: [E * N, T, H]

        s = s.view(E, N, T, -1)
        # s: [E, N, T, H]

        s = s.transpose(1, 2)
        # s: [E, T, N, H]

        return s

    def _forward_latent(self, s):
        hfield = s['hfield']
        # hfield: [E, T, N, 600]

        s = torch.cat([v for k, v in s.items() if k != 'hfield'], dim=-1)
        # s: [E, T, N, *F]

        E, T, N, *F = s.size()

        s = s.transpose(1, 2)
        # s: [E, N, T, *F]

        s = s.reshape(E * N, T, -1)
        # s: [E * N, T, prod(*F)]

        s, (self.hidden_state, self.cell_state) = self.lstm(s, (self.hidden_state, self.cell_state))
        # s: [E * N, T, H]

        s = s.view(E, N, T, -1)
        # s: [E, N, T, H]

        s = s.transpose(1, 2)
        # s: [E, T, N, H]

        return s, self._forward_hfield(hfield)


class Actor_LSTM_v13(BaseLSTM_v13):
    def __init__(self, args):
        super(Actor_LSTM_v13, self).__init__(args)

        self.mean_layer = nn.Linear(args.lstm_hidden_dim, args.action_dim)
        self.mean_layer2 = nn.Linear(args.lstm_hidden_dim, args.action_dim)

        if args.std:
            self.log_std = nn.Parameter(torch.tensor(args.std).log(), requires_grad=False)
        else:
            self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))

        self.log_std2 = nn.Parameter(torch.zeros(1, args.action_dim))

        if args.use_orthogonal_init:
            logging.info("------use_orthogonal_init------")
            if hasattr(self, 'mean_layer2'):
                orthogonal_init(self.mean_layer, gain=0.01)
                orthogonal_init(self.mean_layer2, gain=0.01)

            if hasattr(self, 'std_layer2'):
                orthogonal_init(self.std_layer, gain=0.01)
                orthogonal_init(self.std_layer2, gain=0.01)
        #
        # for param in self.mean_layer.parameters():
        #     param.requires_grad = False

    def forward(self, s):
        s, hfield = self._forward_latent(s)
        # s: [E, T, N, H]

        mean = self.mean_layer(s) + self.mean_layer2(hfield)
        # mean: [E, T, N, action_dim]

        return mean, self.log_std.expand_as(mean).exp()

    def pdf(self, s):
        mean, std = self.forward(s)
        return Normal(mean, std)


class Critic_LSTM_v13(BaseLSTM_v13):
    def __init__(self, args):
        super(Critic_LSTM_v13, self).__init__(args)

        self.value_layer = nn.Sequential(nn.Linear(args.lstm_hidden_dim, 1))

        self.value_layer2 = nn.Sequential(nn.Linear(args.lstm_hidden_dim, 1))

        if args.use_orthogonal_init:
            logging.info("------use_orthogonal_init------")
            if hasattr(self, 'value_layer'):
                for layer in self.value_layer:
                    if isinstance(layer, nn.Linear):
                        orthogonal_init(layer, gain=0.01)

        # for param in self.value_layer.parameters():
        #     param.requires_grad = False

    def forward(self, s):
        s, hfield = self._forward_latent(s)
        # s: [E, T, N, H]

        value = self.value_layer(s) + self.value_layer2(hfield)
        # mean: [E, T, N, 1]

        return value


class Actor_LSTM_v4(BaseLSTM):
    def __init__(self, args):
        super(Actor_LSTM_v4, self).__init__()

        state_dim = args.state_dim

        if 'hfield' in state_dim:
            del state_dim['hfield']

        state_dim = sum([math.prod(shape) for shape in state_dim.values()])

        num_cassie = np.nonzero(args.num_cassie_prob)[0]
        assert len(num_cassie) == 1, \
            'This centralized model only handle fixed num_cassie. num_cassie_prob must be non-zero for just one value.'

        num_cassie = num_cassie[0] + 1

        self.lstm = nn.LSTM(input_size=state_dim, hidden_size=args.lstm_hidden_dim,
                            num_layers=args.lstm_num_layers, batch_first=True)

        self.mean_layer = nn.Linear(args.lstm_hidden_dim, args.action_dim)

        self.composite_layer = nn.Sequential(
            nn.Linear(args.lstm_hidden_dim * num_cassie, args.lstm_hidden_dim * num_cassie),
            nn.ReLU(),
            nn.Linear(args.lstm_hidden_dim * num_cassie, args.lstm_hidden_dim * num_cassie),
        )

        self.composite_residual_mean_layer = nn.Linear(args.lstm_hidden_dim, args.action_dim)

        if args.std:
            self.log_std = nn.Parameter(torch.tensor(args.std).log(), requires_grad=False)
        else:
            self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))

        if args.use_orthogonal_init:
            logging.info("------use_orthogonal_init------")
            if hasattr(self, 'mean_layer'):
                orthogonal_init(self.mean_layer, gain=0.01)

            if hasattr(self, 'std_layer'):
                orthogonal_init(self.std_layer, gain=0.01)

            if hasattr(self, 'composite_layer'):
                for layer in self.composite_layer:
                    if isinstance(layer, nn.Linear):
                        orthogonal_init(layer, gain=0.01)

            if hasattr(self, 'composite_residual_mean_layer'):
                orthogonal_init(self.composite_residual_mean_layer, gain=0.01)

    def forward(self, s):
        s = torch.cat([v for k, v in s.items() if k != 'hfield'], dim=-1)
        # s: [E, T, N, *F]

        E, T, N, *F = s.size()

        s = s.transpose(1, 2)
        # s: [E, N, T, *F]

        s = s.reshape(E * N, T, -1)
        # s: [E * N, T, prod(*F)]

        s, (self.hidden_state, self.cell_state) = self.lstm(s, (self.hidden_state, self.cell_state))
        # s: [E * N, T, H]

        s = s.view(E, N, T, -1)
        # s: [E, N, T, H]

        s = s.transpose(1, 2)
        # s: [E, T, N, H]

        mean = self.mean_layer(s)
        # mean: [E, T, N, action_dim]

        s = s.view(E, T, -1)
        # s: [E, T, N*H]

        s = self.composite_layer(s)
        # s: [E, T, N*H]

        s = s.view(E, T, N, -1)
        # s: [E, T, N, H]

        mean2 = self.composite_residual_mean_layer(s)

        mean = mean + mean2
        # mean: [E, T, N, action_dim]

        return mean, self.log_std.expand_as(mean).exp()

    def pdf(self, s):
        mean, std = self.forward(s)
        return Normal(mean, std)


class Critic_LSTM_v4(BaseLSTM):
    def __init__(self, args):
        super(Critic_LSTM_v4, self).__init__()

        state_dim = args.state_dim

        if 'hfield' in state_dim:
            del state_dim['hfield']

        state_dim = sum([math.prod(shape) for shape in state_dim.values()])

        num_cassie = np.nonzero(args.num_cassie_prob)[0]
        assert len(num_cassie) == 1, \
            'This centralized model only handle fixed num_cassie. num_cassie_prob must be non-zero for just one value.'

        num_cassie = num_cassie[0] + 1

        self.lstm = nn.LSTM(input_size=state_dim, hidden_size=args.lstm_hidden_dim,
                            num_layers=args.lstm_num_layers, batch_first=True)

        self.value_layer = nn.Sequential(
            nn.Linear(args.lstm_hidden_dim, 1)
        )

        self.composite_layer = nn.Sequential(
            nn.Linear(args.lstm_hidden_dim * num_cassie, args.lstm_hidden_dim * num_cassie),
            nn.ReLU(),
            nn.Linear(args.lstm_hidden_dim * num_cassie, args.lstm_hidden_dim * num_cassie),
        )

        self.composite_residual_value_layer = nn.Linear(args.lstm_hidden_dim, 1)

        if args.use_orthogonal_init:
            if hasattr(self, 'composite_layer'):
                for layer in self.composite_layer:
                    if isinstance(layer, nn.Linear):
                        orthogonal_init(layer, gain=0.01)

    def forward(self, s):
        s = torch.cat([v for k, v in s.items() if k != 'hfield'], dim=-1)
        # s: [E, T, N, *F]

        E, T, N, *F = s.size()

        s = s.transpose(1, 2)
        # s: [E, N, T, *F]

        s = s.reshape(E * N, T, -1)
        # s: [E * N, T, prod(*F)]

        s, (self.hidden_state, self.cell_state) = self.lstm(s, (self.hidden_state, self.cell_state))
        # s: [E * N, T, H]

        s = s.view(E, N, T, -1)
        # s: [E, N, T, H]

        s = s.transpose(1, 2)
        # s: [E, T, N, H]

        value = self.value_layer(s)
        # mean: [E, T, N, 1]

        s = s.view(E, T, -1)
        # s: [E, T, N*H]

        s = self.composite_layer(s)
        # s: [E, T, N*H]

        s = s.view(E, T, N, -1)
        # s: [E, T, N, H]

        value = value + self.composite_residual_value_layer(s)
        # mean: [E, T, N, action_dim]

        return value


class BaseLSTM_v5(nn.Module):
    def _init_hidden_state(self, lstm, device=torch.device('cpu'), batch_size=None):
        if batch_size is None:
            hidden_state = torch.zeros(lstm.num_layers, lstm.hidden_size).to(device)
            cell_state = torch.zeros(lstm.num_layers, lstm.hidden_size).to(device)
        else:
            hidden_state = torch.zeros(lstm.num_layers, batch_size, lstm.hidden_size).to(device)
            cell_state = torch.zeros(lstm.num_layers, batch_size, lstm.hidden_size).to(device)

        return hidden_state, cell_state

    def init_hidden_state(self, device=torch.device('cpu'), batch_size=None):
        self.hidden_state, self.cell_state = self._init_hidden_state(self.lstm, device, batch_size)
        self.hidden_state2, self.cell_state2 = self._init_hidden_state(self.composite_lstm, device,
                                                                       batch_size // self.num_cassie)

    def forward(self, s):
        raise NotImplementedError


class Actor_LSTM_v5(BaseLSTM_v5):
    def __init__(self, args):
        super(Actor_LSTM_v5, self).__init__()

        state_dim = args.state_dim

        if 'hfield' in state_dim:
            del state_dim['hfield']

        state_dim = sum([math.prod(shape) for shape in state_dim.values()])

        self.num_cassie = np.nonzero(args.num_cassie_prob)[0]
        assert len(self.num_cassie) == 1, \
            'This centralized model only handle fixed num_cassie. num_cassie_prob must be non-zero for just one value.'

        self.num_cassie = self.num_cassie.item() + 1

        self.lstm = nn.LSTM(input_size=state_dim, hidden_size=args.lstm_hidden_dim,
                            num_layers=args.lstm_num_layers, batch_first=True)

        self.mean_layer = nn.Linear(args.lstm_hidden_dim, args.action_dim)

        self.composite_lstm = nn.LSTM(input_size=state_dim * self.num_cassie,
                                      hidden_size=args.lstm_hidden_dim * self.num_cassie,
                                      num_layers=args.lstm_num_layers, batch_first=True)

        self.composite_mean_layer = nn.Linear(args.lstm_hidden_dim * self.num_cassie, args.action_dim * self.num_cassie)

        if args.std:
            self.log_std = nn.Parameter(torch.tensor(args.std).log(), requires_grad=False)
        else:
            self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))

        if args.use_orthogonal_init:
            logging.info("------use_orthogonal_init------")
            if hasattr(self, 'mean_layer'):
                orthogonal_init(self.mean_layer, gain=0.01)

            if hasattr(self, 'composite_mean_layer'):
                orthogonal_init(self.composite_mean_layer, gain=0.01)

            if hasattr(self, 'std_layer'):
                orthogonal_init(self.std_layer, gain=0.01)

    def forward(self, s):
        s = torch.cat([v for k, v in s.items() if k != 'hfield'], dim=-1)
        # s: [E, T, N, *F]

        E, T, N, *F = s.size()

        s_composite = s.view(E, T, -1)
        # s: [E, T, N*F]

        s = s.transpose(1, 2)
        # s: [E, N, T, *F]

        s = s.reshape(E * N, T, -1)
        # s: [E * N, T, prod(*F)]

        s, (self.hidden_state, self.cell_state) = self.lstm(s, (self.hidden_state, self.cell_state))
        # s: [E * N, T, H]

        s = s.view(E, N, T, -1)
        # s: [E, N, T, H]

        s = s.transpose(1, 2)
        # s: [E, T, N, H]

        mean = self.mean_layer(s)
        # mean: [E, T, N, action_dim]

        # ----------------------------

        s_composite, (self.hidden_state2, self.cell_state2) = \
            self.composite_lstm(s_composite, (self.hidden_state2, self.cell_state2))
        # s: [E, T, N*H]

        mean_composite = self.composite_mean_layer(s_composite)
        # s: [E, T, N * action_dim]

        mean_composite = mean_composite.view(E, T, N, -1)

        mean = mean + mean_composite
        # mean: [E, T, N, action_dim]

        return mean, self.log_std.expand_as(mean).exp()

    def pdf(self, s):
        mean, std = self.forward(s)
        return Normal(mean, std)


class Critic_LSTM_v5(BaseLSTM_v5):
    def __init__(self, args):
        super(Critic_LSTM_v5, self).__init__()

        state_dim = args.state_dim

        if 'hfield' in state_dim:
            del state_dim['hfield']

        state_dim = sum([math.prod(shape) for shape in state_dim.values()])

        self.num_cassie = np.nonzero(args.num_cassie_prob)[0]
        assert len(self.num_cassie) == 1, \
            'This centralized model only handle fixed num_cassie. num_cassie_prob must be non-zero for just one value.'

        self.num_cassie = self.num_cassie.item() + 1

        self.lstm = nn.LSTM(input_size=state_dim, hidden_size=args.lstm_hidden_dim,
                            num_layers=args.lstm_num_layers, batch_first=True)

        self.value_layer = nn.Sequential(
            nn.Linear(args.lstm_hidden_dim, 1)
        )

        self.composite_lstm = nn.LSTM(input_size=state_dim * self.num_cassie,
                                      hidden_size=args.lstm_hidden_dim * self.num_cassie,
                                      num_layers=args.lstm_num_layers, batch_first=True)

        self.composite_value_layer = nn.Linear(args.lstm_hidden_dim * self.num_cassie, self.num_cassie)

        if args.use_orthogonal_init:
            if hasattr(self, 'composite_layer'):
                for layer in self.composite_layer:
                    if isinstance(layer, nn.Linear):
                        orthogonal_init(layer, gain=0.01)

    def forward(self, s):
        s = torch.cat([v for k, v in s.items() if k != 'hfield'], dim=-1)
        # s: [E, T, N, *F]

        E, T, N, *F = s.size()

        s_composite = s.view(E, T, -1)
        # s: [E, T, N*F]

        s = s.transpose(1, 2)
        # s: [E, N, T, *F]

        s = s.reshape(E * N, T, -1)
        # s: [E * N, T, prod(*F)]

        s, (self.hidden_state, self.cell_state) = self.lstm(s, (self.hidden_state, self.cell_state))
        # s: [E * N, T, H]

        s = s.view(E, N, T, -1)
        # s: [E, N, T, H]

        s = s.transpose(1, 2)
        # s: [E, T, N, H]

        value = self.value_layer(s)
        # value: [E, T, N, 1]

        # ----------------------------

        s_composite, (self.hidden_state2, self.cell_state2) = \
            self.composite_lstm(s_composite, (self.hidden_state2, self.cell_state2))
        # s: [E, T, N*F]

        value_composite = self.composite_value_layer(s_composite)
        # s: [E, T, N * 1]

        value_composite = value_composite.view(E, T, N, -1)
        # value: [E, T, N, 1]

        value_composite = value + value_composite

        return value_composite


class BaseTransformer_v10(nn.Module):
    class TransformerBlock(nn.Module):
        def __init__(self, args):
            super(BaseTransformer_v10.TransformerBlock, self).__init__()

            self.positional_encoder = nn.Linear(2, 16)
            self.command_encoder = nn.Linear(4, 16)

            self.encoder_layer = nn.TransformerEncoderLayer(d_model=args.transformer_hidden_dim,
                                                            nhead=args.transformer_num_heads,
                                                            dim_feedforward=args.transformer_dim_feedforward,
                                                            batch_first=True)

            self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=args.transformer_num_layers)

        def forward(self, s, positional_encoding, cmd, src_key_padding_mask, need_weights=False):
            # s: [E, T, N, H=latent_dim]
            # positional_encoding: [E, T, N, 2]
            # positional_encoding: [E, T, 2]
            # src_key_padding_mask: [E, T, N]

            E, T, N = s.size()[:3]

            positional_encoding = self.positional_encoder(positional_encoding)
            # positional_encoding: [E, T, N, 16]

            cmd = self.command_encoder(cmd)
            # cmd: [E, T, N, 16]

            s = torch.cat((s, positional_encoding, cmd), dim=-1)
            # s: [E, T, N, H=transformer_hidden_dim]

            s = s.reshape(E * T, N, -1)
            # s: [E * T, N, H+16+16]

            src_key_padding_mask = src_key_padding_mask.view(E * T, N)
            # src_key_padding_mask: [E * T, N]

            active_seq_mask = (~src_key_padding_mask).any(-1)
            # active_seq_idx: [E * T]

            # Prevents in-place modification for computed gradients
            s_out = s.clone()

            if need_weights:
                s_out[active_seq_mask], attn_weight = self.transformer_encoder(s[active_seq_mask],
                                                                               src_key_padding_mask=
                                                                               src_key_padding_mask[
                                                                                   active_seq_mask],
                                                                               need_weights=need_weights)
            else:
                s_out[active_seq_mask] = self.transformer_encoder(s[active_seq_mask],
                                                                  src_key_padding_mask=src_key_padding_mask[
                                                                      active_seq_mask])
            s = s_out
            # s: [E * T, N, H+16]

            s = s.view(E, T, N, -1)
            # s: [E, T, N, H+16]

            if need_weights:
                return s, attn_weight

            return s

    def __init__(self, args):
        super(BaseTransformer_v10, self).__init__()

        state_dim = deepcopy(args.state_dim)
        state_dim.pop('encoding')
        state_dim.pop('cmd')
        if 'hfield' in state_dim:
            del state_dim['hfield']

        state_dim = sum([math.prod(shape) for shape in state_dim.values()])

        self.lstm = nn.LSTM(input_size=state_dim, hidden_size=args.lstm_hidden_dim,
                            num_layers=args.lstm_num_layers, batch_first=True)

        self.learn_lstm = True
        self.learn_transformer = True

        assert self.learn_lstm or self.learn_transformer, 'At least one of LSTM or Transformer should be used'

        if not self.learn_lstm:
            for param in self.lstm.parameters():
                param.requires_grad = False

        self.transformer_block = self.TransformerBlock(args)

    def init_hidden_state(self, device=torch.device('cpu'), batch_size=None):
        if batch_size is None:
            self.hidden_state = torch.zeros(self.lstm.num_layers, self.lstm.hidden_size).to(device)
            self.cell_state = torch.zeros(self.lstm.num_layers, self.lstm.hidden_size).to(device)
        else:
            self.hidden_state = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device)
            self.cell_state = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device)

    def _forward_latent(self, s, src_key_padding_mask, need_weights=False):
        positional_encoding = s['encoding'][:, :, :, 0]
        # positional_encoding: [E, T, N, 2], selecting first encoding in the state history
        # as encoding is same throughout a window
        cmd = s['cmd'][:, :, :, 0]
        # cmd: [E, T, N, 3]

        s = torch.cat([v for k, v in s.items() if k not in ('encoding', 'cmd', 'hfield')], dim=-1)
        # s: [E, T, N, W, F]

        E, T, N, *F = s.size()

        s = s.transpose(1, 2)
        # s: [E, N, T, W, F]

        s = s.reshape(E * N, T, -1)
        # s: [E * N, T, W * F]

        s, (self.hidden_state, self.cell_state) = self.lstm(s, (self.hidden_state, self.cell_state))
        # s: [E * N, T, H=hidden_size_lstm]

        s = s.view(E, N, T, -1)
        # s: [E, N, T, H]

        s = s.transpose(1, 2)
        # s: [E, T, N, H]

        if self.learn_transformer:
            if need_weights:
                s, attn_weight = self.transformer_block(s, positional_encoding, cmd, src_key_padding_mask, need_weights)
                # s: [E, T, N, H=transformer_hidden_dim]
                # attn_weight: [E, T, N, N]

                return s, attn_weight

            s = self.transformer_block(s, positional_encoding, cmd, src_key_padding_mask)
            # s: [E, T, N, H=transformer_hidden_dim]

            return s

        if need_weights:
            return s, None

        return s

    def forward(self, s, src_key_padding_mask, need_weights=False):
        raise NotImplementedError


class Actor_Transformer_v10(BaseTransformer_v10):
    def __init__(self, args):
        super(Actor_Transformer_v10, self).__init__(args)

        self.mean_layer = nn.Linear(args.transformer_hidden_dim, args.action_dim)

        if args.std:
            self.log_std = nn.Parameter(torch.tensor(args.std).log(), requires_grad=False)
        else:
            self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))

        if args.use_orthogonal_init:
            if hasattr(self, 'mean_layer'):
                orthogonal_init(self.mean_layer, gain=0.01)

            if hasattr(self, 'std_layer'):
                orthogonal_init(self.std_layer, gain=0.01)

    # s: [E=num_episodes/batch_size, T=seq_len, N=num_cassie, *F=*num_features]
    def forward(self, s, src_key_padding_mask, need_weights=False):
        if need_weights:
            s, attn_weight = self._forward_latent(s, src_key_padding_mask, need_weights)
        else:
            s = self._forward_latent(s, src_key_padding_mask, need_weights)
            # s: [E, T, N, H]

        mean = self.mean_layer(s)
        # mean: [E, T, N, action_dim]

        if need_weights:
            return mean, self.log_std.expand_as(mean).exp(), attn_weight

        return mean, self.log_std.expand_as(mean).exp()

    def pdf(self, s, src_key_padding_mask, need_weights=False):

        if need_weights:
            mean, std, attn_weight = self.forward(s, src_key_padding_mask, need_weights)
            return Normal(mean, std), attn_weight

        mean, std = self.forward(s, src_key_padding_mask, need_weights)
        return Normal(mean, std)


class Critic_Transformer_v10(BaseTransformer_v10):
    def __init__(self, args):
        super(Critic_Transformer_v10, self).__init__(args)

        self.value_layer = nn.Linear(args.transformer_hidden_dim, 1)

        if args.use_orthogonal_init:
            if hasattr(self, 'value_layer'):
                orthogonal_init(self.value_layer, gain=0.01)

    def forward(self, s, src_key_padding_mask, need_weights=False):
        s = self._forward_latent(s, src_key_padding_mask, need_weights)
        # s: [E, T, N, H]

        if need_weights:
            value, attn_weight = self.value_layer(s)
            # value: [E, T, N, 1]
            return value, attn_weight

        value = self.value_layer(s)

        return value


class BaseTransformer_v11(nn.Module):
    class AttentionBlock(nn.Module):
        def __init__(self, args):
            super(BaseTransformer_v11.AttentionBlock, self).__init__()
            self.positional_encoder = nn.Linear(2, 16)
            self.command_encoder = nn.Linear(4, 16)

            # Use PyTorch's built-in MultiheadAttention module
            self.multi_head_attention = nn.MultiheadAttention(embed_dim=args.transformer_hidden_dim,
                                                              num_heads=args.transformer_num_heads,
                                                              batch_first=True)

            self.layer_norm1 = nn.LayerNorm(args.transformer_hidden_dim)
            self.layer_norm2 = nn.LayerNorm(args.transformer_hidden_dim)

            self.feed_forward = nn.Sequential(
                nn.Linear(args.transformer_hidden_dim, args.transformer_dim_feedforward),
                nn.ReLU(),
                nn.Linear(args.transformer_dim_feedforward, args.transformer_hidden_dim)
            )

            self.is_decentralized = False

        def forward(self, s, positional_encoding, cmd, src_key_padding_mask, need_weights=False):
            # s: [E, T, N, H=latent_dim]
            # positional_encoding: [E, T, N, 2]
            # cmd: [E, T, N, 4]
            # src_key_padding_mask: [E, T, N]

            E, T, N = s.size()[:3]

            positional_encoding = self.positional_encoder(positional_encoding)
            # positional_encoding: [E, T, N, 16]

            cmd = self.command_encoder(cmd)
            # cmd: [E, T, N, 16]

            s = torch.cat((s, positional_encoding, cmd), dim=-1)
            # s: [E, T, N, H=transformer_hidden_dim]

            s = s.reshape(E * T, N, -1)
            # s: [E * T, N, H+16+16]

            src_key_padding_mask = src_key_padding_mask.view(E * T, N)
            # src_key_padding_mask: [E * T, N]

            active_seq_mask = (~src_key_padding_mask).any(-1)
            # active_seq_idx: [E * T]

            # s_out = s.clone()
            #
            # # Sequence dim
            # num_cassie = s.size(1)
            #
            # if self.is_decentralized:
            #     # Self-attention only
            #     self_attention_mask = torch.ones(num_cassie, num_cassie, dtype=torch.bool, device=s.device)
            #     self_attention_mask.fill_diagonal_(False)
            # else:
            #     self_attention_mask = torch.zeros(num_cassie, num_cassie, dtype=torch.bool, device=s.device)
            #
            # if need_weights:
            #     s_attention, attn_weight = self.multi_head_attention(s[active_seq_mask], s[active_seq_mask],
            #                                                          s[active_seq_mask],
            #                                                          attn_mask=self_attention_mask,
            #                                                          key_padding_mask=src_key_padding_mask[
            #                                                              active_seq_mask], need_weights=need_weights)
            # else:
            #     s_attention, _ = self.multi_head_attention(s[active_seq_mask], s[active_seq_mask], s[active_seq_mask],
            #                                                attn_mask=self_attention_mask,
            #                                                key_padding_mask=src_key_padding_mask[active_seq_mask])
            #
            # # Apply layer normalization after the attention layer
            # s_attention = self.layer_norm1(s[active_seq_mask] + s_attention)
            # # s_attention: [E * T, N, H+16+16]
            #
            # # Feedforward network
            # s_feedforward = self.feed_forward(s_attention)
            # # s_feedforward: [E * T, N, H+16+16]
            #
            # s_out[active_seq_mask] = self.layer_norm2(s_attention + s_feedforward)
            # # s_out: [E * T, N, H+16+16]
            #
            # s = s_out
            # s: [E * T, N, H+16]

            s = s.view(E, T, N, -1)
            # s: [E, T, N, H+16]

            if need_weights:
                return s, attn_weight

            return s

    def __init__(self, args):
        super(BaseTransformer_v11, self).__init__()

        state_dim = deepcopy(args.state_dim)
        state_dim.pop('encoding')
        state_dim.pop('cmd')
        if 'hfield' in state_dim:
            del state_dim['hfield']

        state_dim = sum([math.prod(shape) for shape in state_dim.values()])

        self.lstm = nn.LSTM(input_size=state_dim, hidden_size=args.lstm_hidden_dim,
                            num_layers=args.lstm_num_layers, batch_first=True)

        self.learn_lstm = True
        self.learn_transformer = True

        assert self.learn_lstm or self.learn_transformer, 'At least one of LSTM or Transformer should be used'

        if not self.learn_lstm:
            for param in self.lstm.parameters():
                param.requires_grad = False

        self.transformer_block = self.AttentionBlock(args)

    def init_hidden_state(self, device=torch.device('cpu'), batch_size=None):
        if batch_size is None:
            self.hidden_state = torch.zeros(self.lstm.num_layers, self.lstm.hidden_size).to(device)
            self.cell_state = torch.zeros(self.lstm.num_layers, self.lstm.hidden_size).to(device)
        else:
            self.hidden_state = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device)
            self.cell_state = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device)

    def _forward_latent(self, s, src_key_padding_mask, need_weights=False):
        positional_encoding = s['encoding'][:, :, :, 0]
        # positional_encoding: [E, T, N, 2], selecting first encoding in the state history
        # as encoding is same throughout a window
        cmd = s['cmd'][:, :, :, 0]
        # cmd: [E, T, N, 3]

        s = torch.cat([v for k, v in s.items() if k not in ('encoding', 'cmd', 'hfield')], dim=-1)
        # s: [E, T, N, W, F]

        E, T, N, *F = s.size()

        s = s.transpose(1, 2)
        # s: [E, N, T, W, F]

        s = s.reshape(E * N, T, -1)
        # s: [E * N, T, W * F]

        s, (self.hidden_state, self.cell_state) = self.lstm(s, (self.hidden_state, self.cell_state))
        # s: [E * N, T, H=hidden_size_lstm]

        s = s.view(E, N, T, -1)
        # s: [E, N, T, H]

        s = s.transpose(1, 2)
        # s: [E, T, N, H]

        if self.learn_transformer:
            if need_weights:
                s, attn_weight = self.transformer_block(s, positional_encoding, cmd, src_key_padding_mask, need_weights)
                # s: [E, T, N, H=transformer_hidden_dim]
                # attn_weight: [E, T, N, N]

                return s, attn_weight

            s = self.transformer_block(s, positional_encoding, cmd, src_key_padding_mask)
            # s: [E, T, N, H=transformer_hidden_dim]

            return s

        if need_weights:
            return s, None

        return s

    def forward(self, s, src_key_padding_mask, need_weights=False):
        raise NotImplementedError


class Actor_Transformer_v11(BaseTransformer_v11):
    def __init__(self, args):
        super(Actor_Transformer_v11, self).__init__(args)

        self.mean_layer = nn.Linear(args.transformer_hidden_dim, args.action_dim)

        if args.std:
            self.log_std = nn.Parameter(torch.tensor(args.std).log(), requires_grad=False)
        else:
            self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))

        if args.use_orthogonal_init:
            if hasattr(self, 'mean_layer'):
                orthogonal_init(self.mean_layer, gain=0.01)

            if hasattr(self, 'std_layer'):
                orthogonal_init(self.std_layer, gain=0.01)

    # s: [E=num_episodes/batch_size, T=seq_len, N=num_cassie, *F=*num_features]
    def forward(self, s, src_key_padding_mask, need_weights=False):
        if need_weights:
            s, attn_weight = self._forward_latent(s, src_key_padding_mask, need_weights)
        else:
            s = self._forward_latent(s, src_key_padding_mask, need_weights)
            # s: [E, T, N, H]

        mean = self.mean_layer(s)
        # mean: [E, T, N, action_dim]

        if need_weights:
            return mean, self.log_std.expand_as(mean).exp(), attn_weight

        return mean, self.log_std.expand_as(mean).exp()

    def pdf(self, s, src_key_padding_mask, need_weights=False):

        if need_weights:
            mean, std, attn_weight = self.forward(s, src_key_padding_mask, need_weights)
            return Normal(mean, std), attn_weight

        mean, std = self.forward(s, src_key_padding_mask, need_weights)
        return Normal(mean, std)


class Critic_Transformer_v11(BaseTransformer_v11):
    def __init__(self, args):
        super(Critic_Transformer_v11, self).__init__(args)

        self.value_layer = nn.Linear(args.transformer_hidden_dim, 1)

        if args.use_orthogonal_init:
            if hasattr(self, 'value_layer'):
                orthogonal_init(self.value_layer, gain=0.01)

    def forward(self, s, src_key_padding_mask, need_weights=False):
        s = self._forward_latent(s, src_key_padding_mask, need_weights)
        # s: [E, T, N, H]

        if need_weights:
            value, attn_weight = self.value_layer(s)
            # value: [E, T, N, 1]
            return value, attn_weight

        value = self.value_layer(s)

        return value


class BaseTransformer_v12(nn.Module):
    class AttentionBlock(nn.Module):
        def __init__(self, args):
            super(BaseTransformer_v12.AttentionBlock, self).__init__()

            # Use PyTorch's built-in MultiheadAttention module
            self.multi_head_attention = nn.MultiheadAttention(embed_dim=args.transformer_hidden_dim,
                                                              num_heads=args.transformer_num_heads,
                                                              batch_first=True)

            self.feed_forward = nn.Sequential(
                nn.Linear(args.transformer_hidden_dim, args.transformer_dim_feedforward),
                nn.ReLU(),
                nn.Linear(args.transformer_dim_feedforward, args.transformer_hidden_dim)
            )

        def forward(self, s, src_key_padding_mask, need_weights=False):
            # s: [E, T, N, H=latent_dim]
            # positional_encoding: [E, T, N, 2]
            # cmd: [E, T, N, 4]
            # src_key_padding_mask: [E, T, N]

            E, T, N = s.size()[:3]

            s = s.reshape(E * T, N, -1)
            # s: [E * T, N, H]

            src_key_padding_mask = src_key_padding_mask.view(E * T, N)
            # src_key_padding_mask: [E * T, N]

            active_seq_mask = (~src_key_padding_mask).any(-1)
            # active_seq_idx: [E * T]

            s_out = s.clone()

            # seq_len = s.size(1)
            # self_attention_mask = torch.full((seq_len, seq_len), float('-inf'))
            # self_attention_mask.fill_diagonal_(0)

            if need_weights:
                s_attention, attn_weight = self.multi_head_attention(s[active_seq_mask], s[active_seq_mask],
                                                                     s[active_seq_mask],
                                                                     # attn_mask=self_attention_mask,
                                                                     key_padding_mask=src_key_padding_mask[
                                                                         active_seq_mask], need_weights=need_weights)
            else:
                s_attention, _ = self.multi_head_attention(s[active_seq_mask], s[active_seq_mask], s[active_seq_mask],
                                                           key_padding_mask=src_key_padding_mask[active_seq_mask])

            # print('atn',s_attention.min(), s_attention.max(), s_attention.mean(), s_attention.std())
            # print(s[active_seq_mask].min(), s[active_seq_mask].max(), s[active_seq_mask].mean(), s[active_seq_mask].std())
            s_out[active_seq_mask] = s_attention + s[active_seq_mask]
            # s_out: [E * T, N, H+16+16]

            s = s_out
            # s: [E * T, N, H+16]

            s = s.view(E, T, N, -1)
            # s: [E, T, N, H+16]

            if need_weights:
                return s, attn_weight

            return s

    def __init__(self, args):
        super(BaseTransformer_v12, self).__init__()

        state_dim = deepcopy(args.state_dim)
        # state_dim.pop('encoding')
        # state_dim.pop('cmd')
        if 'hfield' in state_dim:
            del state_dim['hfield']

        state_dim = sum([math.prod(shape) for shape in state_dim.values()])

        self.lstm = nn.LSTM(input_size=state_dim, hidden_size=args.lstm_hidden_dim,
                            num_layers=args.lstm_num_layers, batch_first=True)

        self.learn_lstm = False
        self.learn_transformer = True

        assert self.learn_lstm or self.learn_transformer, 'At least one of LSTM or Transformer should be used'

        if not self.learn_lstm:
            for param in self.lstm.parameters():
                param.requires_grad = False

        self.attention_block = self.AttentionBlock(args)

    def init_hidden_state(self, device=torch.device('cpu'), batch_size=None):
        if batch_size is None:
            self.hidden_state = torch.zeros(self.lstm.num_layers, self.lstm.hidden_size).to(device)
            self.cell_state = torch.zeros(self.lstm.num_layers, self.lstm.hidden_size).to(device)
        else:
            self.hidden_state = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device)
            self.cell_state = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device)

    def _forward_latent(self, s, src_key_padding_mask, need_weights=False):
        # positional_encoding = s['encoding'][:, :, :, 0]
        # positional_encoding: [E, T, N, 2], selecting first encoding in the state history
        # as encoding is same throughout a window
        # cmd = s['cmd'][:, :, :, 0]
        # cmd: [E, T, N, 3]

        s = torch.cat([v for k, v in s.items() if k not in ('hfield',)], dim=-1)
        # s: [E, T, N, W, F]

        E, T, N, *F = s.size()

        s = s.transpose(1, 2)
        # s: [E, N, T, W, F]

        s = s.reshape(E * N, T, -1)
        # s: [E * N, T, W * F]

        s, (self.hidden_state, self.cell_state) = self.lstm(s, (self.hidden_state, self.cell_state))
        # s: [E * N, T, H=hidden_size_lstm]

        s = s.view(E, N, T, -1)
        # s: [E, N, T, H]

        s = s.transpose(1, 2)
        # s: [E, T, N, H]

        if self.learn_transformer:
            if need_weights:
                s, attn_weight = self.attention_block(s, src_key_padding_mask, need_weights)
                # s: [E, T, N, H=transformer_hidden_dim]
                # attn_weight: [E, T, N, N]

                return s, attn_weight

            s = self.attention_block(s, src_key_padding_mask)
            # s: [E, T, N, H=transformer_hidden_dim]

            return s

        if need_weights:
            return s, None

        return s

    def forward(self, s, src_key_padding_mask, need_weights=False):
        raise NotImplementedError


class Actor_Transformer_v12(BaseTransformer_v12):
    def __init__(self, args):
        super(Actor_Transformer_v12, self).__init__(args)

        self.mean_layer = nn.Linear(args.transformer_hidden_dim, args.action_dim)

        if args.std:
            self.log_std = nn.Parameter(torch.tensor(args.std).log(), requires_grad=False)
        else:
            self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))

        if args.use_orthogonal_init:
            if hasattr(self, 'mean_layer'):
                orthogonal_init(self.mean_layer, gain=0.01)

            if hasattr(self, 'std_layer'):
                orthogonal_init(self.std_layer, gain=0.01)

    # s: [E=num_episodes/batch_size, T=seq_len, N=num_cassie, *F=*num_features]
    def forward(self, s, src_key_padding_mask, need_weights=False):
        if need_weights:
            s, attn_weight = self._forward_latent(s, src_key_padding_mask, need_weights)
        else:
            s = self._forward_latent(s, src_key_padding_mask, need_weights)
            # s: [E, T, N, H]

        mean = self.mean_layer(s)
        # mean: [E, T, N, action_dim]

        if need_weights:
            return mean, self.log_std.expand_as(mean).exp(), attn_weight

        return mean, self.log_std.expand_as(mean).exp()

    def pdf(self, s, src_key_padding_mask, need_weights=False):

        if need_weights:
            mean, std, attn_weight = self.forward(s, src_key_padding_mask, need_weights)
            return Normal(mean, std), attn_weight

        mean, std = self.forward(s, src_key_padding_mask, need_weights)
        return Normal(mean, std)


class Critic_Transformer_v12(BaseTransformer_v12):
    def __init__(self, args):
        super(Critic_Transformer_v12, self).__init__(args)

        self.value_layer = nn.Linear(args.transformer_hidden_dim, 1)

        if args.use_orthogonal_init:
            if hasattr(self, 'value_layer'):
                orthogonal_init(self.value_layer, gain=0.01)

    def forward(self, s, src_key_padding_mask, need_weights=False):
        s = self._forward_latent(s, src_key_padding_mask, need_weights)
        # s: [E, T, N, H]

        if need_weights:
            value, attn_weight = self.value_layer(s)
            # value: [E, T, N, 1]
            return value, attn_weight

        value = self.value_layer(s)

        return value


class BaseTransformer_v13(nn.Module):

    def load_state_dict(self, state_dict: Mapping[str, Any],
                        strict: bool = True, assign: bool = False):
        pass

    class TemporalEncoding(nn.Module):

        def __init__(self, state_dim: int, max_len: int, dropout: float = 0.1):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)

            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, state_dim, 2) * (-math.log(10000.0) / state_dim))
            pe = torch.zeros(1, 1, 1, max_len, state_dim)
            pe[0, 0, 0, :, 0::2] = torch.sin(position * div_term)
            if state_dim % 2 == 0:
                pe[0, 0, 0, :, 1::2] = torch.cos(position * div_term)
            else:
                pe[0, 0, 0, :, 1::2] = torch.cos(position * div_term[:-1])
            self.register_buffer('pe', pe)

        # x: [E, T, N, W, F]
        def forward(self, x):
            # print('x.size()', x.size())
            x = x + self.pe[0, 0, 0, :x.size(3)]
            return self.dropout(x)

    def __init__(self, args):
        super(BaseTransformer_v13, self).__init__()

        state_dim = deepcopy(args.state_dim)
        # state_dim.pop('encoding')
        # state_dim.pop('cmd')
        if 'hfield' in state_dim:
            del state_dim['hfield']

        state_dim = sum([shape[-1] for shape in state_dim.values()])
        seq_len = args.state_history_size

        self.fc1 = nn.Sequential(
            nn.Linear(state_dim, args.transformer_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.transformer_hidden_dim, args.transformer_hidden_dim),
        )

        self.temporal_encoding = self.TemporalEncoding(state_dim=args.transformer_hidden_dim, max_len=seq_len)

        _encoder_layer = nn.TransformerEncoderLayer(d_model=args.transformer_hidden_dim,
                                                    nhead=args.transformer_num_heads,
                                                    dropout=0.0,
                                                    dim_feedforward=args.transformer_dim_feedforward,
                                                    batch_first=True)

        self.transformer_encoder = nn.TransformerEncoder(_encoder_layer, num_layers=args.transformer_num_layers)

        self.fc2 = nn.Sequential(
            nn.Linear(seq_len, seq_len),
            nn.ReLU(),
            nn.Linear(seq_len, 1),
        )

    def init_hidden_state(self, device=torch.device('cpu'), batch_size=None):
        # no lstm
        pass

    def _forward_latent(self, s, src_key_padding_mask, need_weights=False):
        s = torch.cat([v for k, v in s.items() if k not in ('hfield',)], dim=-1)
        # s: [E, T, N, W, F]

        E, T, N, W, F = s.size()

        s = self.fc1(s)
        # s: [E, T, N, W, H]

        H = s.size(-1)

        s = self.temporal_encoding(s)
        # s: [E, T, N, W, H]

        s = s.view(E * T, -1, H)
        # s: [E * T, N * W, H]

        src_key_padding_mask = \
            src_key_padding_mask.unsqueeze(-1).expand(-1, -1, -1, W).reshape(E * T, -1)
        # [E * T, N * W]

        active_seq_mask = (~src_key_padding_mask).any(1)
        # [E * T]

        s_ = s.clone()

        if need_weights:
            s_[active_seq_mask], attn_weight = self.transformer_encoder(s_[active_seq_mask],
                                                                        src_key_padding_mask=src_key_padding_mask[
                                                                            active_seq_mask],
                                                                        need_weights=need_weights)
            # s: [B, N * W, H]
            # attn_weight: [B, N * W, N * W]
        else:
            s_[active_seq_mask] = self.transformer_encoder(s_[active_seq_mask],
                                                           src_key_padding_mask=src_key_padding_mask[active_seq_mask])
        # shape:  [E * T, N * W, H]

        s = s_

        s = s.view(E, T, N, -1, H)
        # s: [E, T, N, W, H]

        s = s.transpose(3, 4)
        # s: [E, T, N, H, W]

        s = self.fc2(s).squeeze(-1)
        # s: [E, T, N, H]

        if need_weights:
            return s, attn_weight

        return s

    def forward(self, s, src_key_padding_mask, need_weights=False):
        raise NotImplementedError


class Actor_Transformer_v13(BaseTransformer_v13):
    def __init__(self, args):
        super(Actor_Transformer_v13, self).__init__(args)

        self.mean_layer = nn.Linear(args.transformer_hidden_dim, args.action_dim)

        if args.std:
            self.log_std = nn.Parameter(torch.tensor(args.std).log(), requires_grad=False)
        else:
            self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))

        if args.use_orthogonal_init:
            if hasattr(self, 'mean_layer'):
                orthogonal_init(self.mean_layer, gain=0.01)

            if hasattr(self, 'std_layer'):
                orthogonal_init(self.std_layer, gain=0.01)

    # s: [E=num_episodes/batch_size, T=seq_len, N=num_cassie, *F=*num_features]
    def forward(self, s, src_key_padding_mask, need_weights=False):
        if need_weights:
            s, attn_weight = self._forward_latent(s, src_key_padding_mask, need_weights)
        else:
            s = self._forward_latent(s, src_key_padding_mask, need_weights)
            # s: [E, T, N, H]

        mean = self.mean_layer(s)
        # mean: [E, T, N, action_dim]

        if need_weights:
            return mean, self.log_std.expand_as(mean).exp(), attn_weight

        return mean, self.log_std.expand_as(mean).exp()

    def pdf(self, s, src_key_padding_mask, need_weights=False):

        if need_weights:
            mean, std, attn_weight = self.forward(s, src_key_padding_mask, need_weights)
            return Normal(mean, std), attn_weight

        mean, std = self.forward(s, src_key_padding_mask, need_weights)
        return Normal(mean, std)


class Critic_Transformer_v13(BaseTransformer_v13):
    def __init__(self, args):
        super(Critic_Transformer_v13, self).__init__(args)

        self.value_layer = nn.Linear(args.transformer_hidden_dim, 1)

        if args.use_orthogonal_init:
            if hasattr(self, 'value_layer'):
                orthogonal_init(self.value_layer, gain=0.01)

    def forward(self, s, src_key_padding_mask, need_weights=False):
        s = self._forward_latent(s, src_key_padding_mask, need_weights)
        # s: [E, T, N, H]

        if need_weights:
            value, attn_weight = self.value_layer(s)
            # value: [E, T, N, 1]
            return value, attn_weight

        value = self.value_layer(s)

        return value


class BaseTransformer_v14(nn.Module):
    class AttentionBlock(nn.Module):
        def __init__(self, args):
            super(BaseTransformer_v14.AttentionBlock, self).__init__()

            self.positional_encoder = nn.Linear(2, 16)

            self.multi_head_attention = nn.MultiheadAttention(embed_dim=args.transformer_hidden_dim,
                                                              num_heads=args.transformer_num_heads,
                                                              batch_first=True)

            self.compressor = nn.Sequential(
                nn.Linear(args.transformer_hidden_dim, args.transformer_dim_feedforward),
                nn.ReLU(),
                nn.Linear(args.transformer_dim_feedforward, args.lstm_hidden_dim)
            )

        def forward(self, s_, positional_encoding, src_key_padding_mask, need_weights=False):
            # s: [E, T, N, H_lstm]
            # positional_encoding: [E, T, N, 2]

            positional_encoding = self.positional_encoder(positional_encoding)
            # positional_encoding: [E, T, N, 16]

            s = torch.cat((s_, positional_encoding), dim=-1)
            # s: [E, T, N, H_tx]

            E, T, N = s.size()[:3]

            s = s.reshape(E * T, N, -1)
            # s: [E * T, N, H_tx]

            s_ = s_.reshape(E * T, N, -1)
            # s_: [E * T, N, H_lstm]

            src_key_padding_mask = src_key_padding_mask.view(E * T, N)
            # src_key_padding_mask: [E * T, N]

            active_seq_mask = (~src_key_padding_mask).any(-1)
            # active_seq_idx: [E * T]

            s_out = s_.clone()
            # shape: [E * T, N, H_tx]

            # seq_len = s.size(1)
            # self_attention_mask = torch.full((seq_len, seq_len), float('-inf'))
            # self_attention_mask.fill_diagonal_(0)

            if need_weights:
                s_attention, attn_weight = self.multi_head_attention(s[active_seq_mask], s[active_seq_mask],
                                                                     s[active_seq_mask],
                                                                     # attn_mask=self_attention_mask,
                                                                     key_padding_mask=src_key_padding_mask[
                                                                         active_seq_mask], need_weights=need_weights)
            else:
                s_attention, _ = self.multi_head_attention(s[active_seq_mask], s[active_seq_mask], s[active_seq_mask],
                                                           key_padding_mask=src_key_padding_mask[active_seq_mask])

                # shape: [E * T, N, H_tx]

            s_attention = self.compressor(s_attention)
            # s_attention: [E * T, N, H_lstm]

            s_out[active_seq_mask] = s_attention + s_[active_seq_mask]
            # s_out: [E * T, N, H_lstm]

            s = s_out
            # s: [E * T, N, H_lstm]

            s = s.view(E, T, N, -1)
            # s: [E, T, N, H_lstm]

            if need_weights:
                return s, attn_weight

            return s

    # def load_state_dict(self, state_dict: Mapping[str, Any],
    #                     strict: bool = True, assign: bool = False):
    #     super().load_state_dict(state_dict, False, assign)

    def __init__(self, args):
        super(BaseTransformer_v14, self).__init__()

        state_dim = deepcopy(args.state_dim)
        # state_dim.pop('encoding')
        # state_dim.pop('cmd')
        if 'hfield' in state_dim:
            del state_dim['hfield']

        state_dim = sum([math.prod(shape) for shape in state_dim.values()])

        self.lstm = nn.LSTM(input_size=state_dim, hidden_size=args.lstm_hidden_dim,
                            num_layers=args.lstm_num_layers, batch_first=True)

        self.learn_lstm = True
        self.learn_transformer = True

        assert self.learn_lstm or self.learn_transformer, 'At least one of LSTM or Transformer should be used'

        if not self.learn_lstm:
            for param in self.lstm.parameters():
                param.requires_grad = False

        self.attention_block = self.AttentionBlock(args)

    def init_hidden_state(self, device=torch.device('cpu'), batch_size=None):
        if batch_size is None:
            self.hidden_state = torch.zeros(self.lstm.num_layers, self.lstm.hidden_size).to(device)
            self.cell_state = torch.zeros(self.lstm.num_layers, self.lstm.hidden_size).to(device)
        else:
            self.hidden_state = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device)
            self.cell_state = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device)

    def _forward_latent(self, s, src_key_padding_mask, need_weights=False):
        positional_encoding = s['encoding'][:, :, :, 0]

        # positional_encoding: [E, T, N, 2], selecting first encoding in the state history

        # as encoding is same throughout a window
        # cmd = s['cmd'][:, :, :, 0]
        # cmd: [E, T, N, 3]

        s = torch.cat([torch.zeros_like(v) if k == 'encoding' else v for k, v in s.items() if k not in ('hfield',)],
                      dim=-1)
        # s: [E, T, N, W, F]

        E, T, N, *F = s.size()

        s = s.transpose(1, 2)
        # s: [E, N, T, W, F]

        s = s.reshape(E * N, T, -1)
        # s: [E * N, T, W * F]

        s, (self.hidden_state, self.cell_state) = self.lstm(s, (self.hidden_state, self.cell_state))
        # s: [E * N, T, H=hidden_size_lstm]

        s = s.view(E, N, T, -1)
        # s: [E, N, T, H_lstm]

        s = s.transpose(1, 2)
        # s: [E, T, N, H_lstm]

        if self.learn_transformer:
            if need_weights:
                s, attn_weight = self.attention_block(s, positional_encoding, src_key_padding_mask, need_weights)
                # s: [E, T, N, H_lstm]
                # attn_weight: [E, T, N, N]

                return s, attn_weight

            s = self.attention_block(s, positional_encoding, src_key_padding_mask)
            # s: [E, T, N, H_lstm]

            return s

        if need_weights:
            return s, None

        return s

    def forward(self, s, src_key_padding_mask, need_weights=False):
        raise NotImplementedError


class Actor_Transformer_v14(BaseTransformer_v14):
    def __init__(self, args):
        super(Actor_Transformer_v14, self).__init__(args)

        self.mean_layer = nn.Linear(args.lstm_hidden_dim, args.action_dim)

        if args.std:
            self.log_std = nn.Parameter(torch.tensor(args.std).log(), requires_grad=False)
        else:
            self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))

        if args.use_orthogonal_init:
            if hasattr(self, 'mean_layer'):
                orthogonal_init(self.mean_layer, gain=0.01)

            if hasattr(self, 'std_layer'):
                orthogonal_init(self.std_layer, gain=0.01)

    # s: [E=num_episodes/batch_size, T=seq_len, N=num_cassie, *F=*num_features]
    def forward(self, s, src_key_padding_mask, need_weights=False):
        if need_weights:
            s, attn_weight = self._forward_latent(s, src_key_padding_mask, need_weights)
        else:
            s = self._forward_latent(s, src_key_padding_mask, need_weights)
            # s: [E, T, N, H_lstm]

        mean = self.mean_layer(s)
        # mean: [E, T, N, action_dim]

        if need_weights:
            return mean, self.log_std.expand_as(mean).exp(), attn_weight

        return mean, self.log_std.expand_as(mean).exp()

    def pdf(self, s, src_key_padding_mask, need_weights=False):

        if need_weights:
            mean, std, attn_weight = self.forward(s, src_key_padding_mask, need_weights)
            return Normal(mean, std), attn_weight

        mean, std = self.forward(s, src_key_padding_mask, need_weights)
        return Normal(mean, std)


class Critic_Transformer_v14(BaseTransformer_v14):
    def __init__(self, args):
        super(Critic_Transformer_v14, self).__init__(args)

        self.value_layer = nn.Linear(args.lstm_hidden_dim, 1)

        if args.use_orthogonal_init:
            if hasattr(self, 'value_layer'):
                orthogonal_init(self.value_layer, gain=0.01)

    def forward(self, s, src_key_padding_mask, need_weights=False):
        s = self._forward_latent(s, src_key_padding_mask, need_weights)
        # s: [E, T, N, H]

        if need_weights:
            value, attn_weight = self.value_layer(s)
            # value: [E, T, N, 1]
            return value, attn_weight

        value = self.value_layer(s)

        return value
