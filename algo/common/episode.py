import datetime
from collections import OrderedDict

import torch


class Episode:
    def __init__(self, seq_len, num_cassie, state_dim, action_dim, store_log_prob=False, store_value=False):

        self.episode = dict(
            a=torch.zeros(seq_len, num_cassie, action_dim, dtype=torch.float32),
            r=torch.zeros(seq_len, num_cassie, dtype=torch.float32),
            dw=torch.ones(seq_len, num_cassie, dtype=torch.bool),
            active=torch.zeros(seq_len, num_cassie, dtype=torch.bool),
        )

        if store_log_prob:
            self.episode['log_prob'] = torch.zeros(seq_len, num_cassie, dtype=torch.float32)

        if store_value:
            self.episode['v'] = torch.zeros(seq_len + 1, num_cassie, dtype=torch.float32)

        self.episode['s'] = OrderedDict()

        for k, shape in state_dim.items():
            # Add one to seq_len to store the last state
            self.episode['s'][k] = torch.zeros(seq_len + 1, num_cassie, *shape, dtype=torch.float32)

        self.curr_len = 0

        self.meta_data = None

    def get_transition_keys(self):
        return list(self.episode.keys())

    def get_state_keys(self):
        return list(self.episode['s'].keys())

    def __getitem__(self, item):
        return self.episode[item]

    def __setitem__(self, key, value):
        self.episode[key] = value

    def __len__(self):
        return self.curr_len

    def store_transition(self, kv):
        for k, v in kv.items():
            if k == 's':
                for k in v.keys():
                    self.episode['s'][k][self.curr_len] = v[k]
            else:
                self.episode[k][self.curr_len] = v

    def step(self):
        self.curr_len += 1

    def end(self, meta_data=None):
        for k in self.episode.keys():
            if k == 's':
                for k in self.episode['s'].keys():
                    self.episode['s'][k] = self.episode['s'][k][:self.curr_len + 1]#.contiguous()
            elif k == 'v':
                self.episode[k] = self.episode[k][:self.curr_len + 1]#.contiguous()
            else:
                self.episode[k] = self.episode[k][:self.curr_len]#.contiguous()

        # for k in self.episode.keys():
        #     if k == 's':
        #         for k_ in self.episode['s'].keys():
        #             print(k, k_, self.episode['s'][k_].shape)
        #     else:
        #         print(k, self.episode[k].shape)

        self.meta_data = meta_data

    def __lt__(self, other):
        if isinstance(other, Episode) and self.meta_data is not None and other.meta_data is not None:
            return self.meta_data['iteration'] < other.meta_data['iteration']
        return NotImplemented

    # Optionally, you can implement other comparison operators
    def __le__(self, other):
        if isinstance(other, Episode) and self.meta_data is not None and other.meta_data is not None:
            return self.meta_data['iteration'] <= other.meta_data['iteration']
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, Episode) and self.meta_data is not None and other.meta_data is not None:
            return self.meta_data['iteration'] == other.meta_data['iteration']
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, Episode) and self.meta_data is not None and other.meta_data is not None:
            return self.meta_data['iteration'] != other.meta_data['iteration']
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, Episode) and self.meta_data is not None and other.meta_data is not None:
            return self.meta_data['iteration'] > other.meta_data['iteration']
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, Episode) and self.meta_data is not None and other.meta_data is not None:
            return self.meta_data['iteration'] >= other.meta_data['iteration']
        return NotImplemented
