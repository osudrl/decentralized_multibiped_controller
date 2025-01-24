import argparse
import glob
import importlib
import logging
import lzma
import os
import pickle
import re
import shutil
import subprocess
from collections import OrderedDict, defaultdict
from copy import deepcopy
from itertools import count

import ray
import torch
import wandb
from torch.nn.functional import pad
import yaml

from algo.common.episode import Episode
from algo.common.normalization import RewardScaling

logging.basicConfig(level=logging.INFO)


def normalize(src, src_min, src_max, trg_min, trg_max):
    trg = (src - src_min) * (trg_max - trg_min) / (src_max - src_min) + trg_min

    trg = np.clip(trg, trg_min, trg_max)

    return trg


def execute_command(command):
    if logging.getLogger().isEnabledFor(logging.DEBUG):
        process = subprocess.Popen(command, shell=True, preexec_fn=os.setsid)
    else:
        process = subprocess.Popen(command, shell=True, preexec_fn=os.setsid,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Wait for the process to complete or get cancelled
    return process


def transfer_files(source_dir, dest_dir,
                   # options='--ignore-existing --whole-file',
                   options='',
                   debug=logging.getLogger().isEnabledFor(logging.DEBUG)):
    if source_dir == dest_dir:
        logging.warning(f'Source and destination directories are the same: {source_dir}')
        return

    command = f"rsync -a{'v' if debug else ''} {options or ''} {source_dir}/ {dest_dir}/"

    if debug:
        logging.info(f'DEBUG: Executing command: {command}')

    return execute_command(command)


def init_logger(args, agent):
    iterations = 0
    total_steps = 0
    trajectory_count = 0

    # 4 ways to initialize wandb
    # 1. Parent run is not given, previous run is not given -> Start a new run from scratch
    # 2. Parent run is given, previous run is not given -> Create a new run resumed but detached from parent
    # 3. Parent run is not given, previous run is given -> Resume previous run attached to same parent
    # 4. Parent run is given, previous run is given -> Start a new run from previous run attached to same parent

    if args.parent_run is None and args.previous_run is None:
        run = wandb.init(
            project=args.project_name,
            name=args.run_name,
            mode=args.wandb_mode,
            config={**args.__dict__, 'parent_run': args.run_name},
            id=args.run_name.replace(':', '_'),
        )
    elif args.previous_run is None:
        wandb.login()

        run = wandb.Api().run(os.path.join(args.project_name, args.parent_run.replace(':', '_')))

        logging.info(f'Checkpoint loaded from: {args.parent_run}')

        if args.previous_checkpoint:
            checkpoint_name = f'checkpoints/checkpoint-{args.parent_run}-{args.previous_checkpoint}.pt'
        else:
            checkpoint_name = f'checkpoints/checkpoint-{args.parent_run}.pt'

        run.file(name=checkpoint_name).download(replace=True)

        with open(checkpoint_name, 'rb') as r:
            checkpoint = torch.load(r, map_location=agent.device)

        # Create new run
        run = wandb.init(
            project=args.project_name,
            name=args.run_name,
            config={**args.__dict__, 'parent_run': args.parent_run},  # , 'previous_config': run.config},
            id=args.run_name.replace(':', '_'),
        )

        # Since we start a new run detached from parent, we don't load run state
        total_steps = 0
        trajectory_count = 0
        iterations = 0
        # agent.actor.load_state_dict(checkpoint['actor_state_dict'], strict=False)
        # agent.critic.load_state_dict(checkpoint['critic_state_dict'], strict=False)
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        agent.optimizer_actor.load_state_dict(checkpoint['optimizer_actor_state_dict'])
        agent.optimizer_critic.load_state_dict(checkpoint['optimizer_critic_state_dict'])
    elif args.parent_run is None:
        run_ = wandb.Api().run(os.path.join(args.project_name, args.previous_run.replace(':', '_')))

        run = wandb.init(
            project=args.project_name,
            resume='allow',
            config={**args.__dict__, 'parent_run': run_.config['run_name']},  # , 'previous_config': run_.config},
            id=args.previous_run.replace(':', '_'),
        )

        if run.resumed:
            logging.info(f'Checkpoint loaded from: {args.previous_run}')

            if args.previous_checkpoint:
                checkpoint_name = f'checkpoints/checkpoint-{args.previous_run}-{args.previous_checkpoint}.pt'
            else:
                checkpoint_name = f'checkpoints/checkpoint-{args.previous_run}.pt'

            try:
                run_.file(name=checkpoint_name).download(replace=True)

                with open(checkpoint_name, 'rb') as r:
                    checkpoint = torch.load(r, map_location=agent.device)

                logging.info(f'Resuming from the run: {run.name} ({run.id})')
                total_steps = checkpoint['total_steps']
                trajectory_count = checkpoint['trajectory_count']
                iterations = checkpoint.get('iterations', 0)
                agent.actor.load_state_dict(checkpoint['actor_state_dict'])
                agent.critic.load_state_dict(checkpoint['critic_state_dict'])
                agent.optimizer_actor.load_state_dict(checkpoint['optimizer_actor_state_dict'])
                agent.optimizer_critic.load_state_dict(checkpoint['optimizer_critic_state_dict'])
            except Exception as e:
                total_steps = 0
                trajectory_count = 0
                iterations = 0
                logging.error(f'Run: {args.previous_run} cannot load checkpoint')
                # raise e
        else:
            logging.error(f'Run: {args.previous_run} did not resume')
            raise Exception(f'Run: {args.previous_run} did not resume')
    else:
        wandb.login()

        run_ = wandb.Api().run(os.path.join(args.project_name, args.previous_run.replace(':', '_')))

        logging.info(f'Checkpoint loaded from: {args.previous_run}')

        if args.previous_checkpoint:
            checkpoint_name = f'checkpoints/checkpoint-{args.previous_run}-{args.previous_checkpoint}.pt'
        else:
            checkpoint_name = f'checkpoints/checkpoint-{args.previous_run}.pt'

        run_.file(name=checkpoint_name).download(replace=True)

        with open(checkpoint_name, 'rb') as r:
            checkpoint = torch.load(r, map_location=agent.device)

        # Create new run
        run = wandb.init(
            project=args.project_name,
            name=args.run_name,
            config={**args.__dict__, 'parent_run': args.parent_run},  # , 'previous_config': run_.config},
            id=args.run_name.replace(':', '_'),
        )

        actor_state_dict = checkpoint['actor_state_dict']
        critic_state_dict = checkpoint['critic_state_dict']

        # # Remove the keys from the state_dict
        # for key in list(actor_state_dict.keys()):
        #     if key.startswith('transformer_block'):
        #         actor_state_dict.pop(key)
        #
        # for key in list(critic_state_dict.keys()):
        #     if key.startswith('transformer_block'):
        #         critic_state_dict.pop(key)

        # Load the modified state_dict into the model
        total_steps = checkpoint['total_steps']
        trajectory_count = checkpoint['trajectory_count']
        iterations = checkpoint.get('iterations', 0)
        agent.actor.load_state_dict(actor_state_dict)
        agent.critic.load_state_dict(critic_state_dict)
        agent.optimizer_actor.load_state_dict(checkpoint['optimizer_actor_state_dict'])
        agent.optimizer_critic.load_state_dict(checkpoint['optimizer_critic_state_dict'])

    # Save files to wandb
    cwd = os.getcwd()

    exclude = {'checkpoints', 'saved_models', 'wandb', '.idea', '.git', 'pretrained_models', 'trained_models',
               'offline_data', 'old_files'}

    inc_exts = ['*.py', '*.yaml', '*.yml', '*.json', 'cassiepede[1-3].xml', '*.sh']

    dirs = [d for d in os.listdir(cwd) if d not in exclude and os.path.isdir(os.path.join(cwd, d))]

    # Process files in subdirectories recursively
    for d in dirs:

        base_paths = [os.path.join(cwd, d, '**', ext) for ext in inc_exts]

        for base_path in base_paths:
            for file in glob.glob(base_path, recursive=True):
                file_path = os.path.relpath(file, start=cwd)
                run.save(file_path, policy='now')

    # Process files in current directory
    base_paths = [os.path.join(cwd, ext) for ext in inc_exts]
    for base_path in base_paths:
        for file in glob.glob(base_path):
            file_path = os.path.relpath(file, start=cwd)
            run.save(file_path, policy='now')

    return run, iterations, total_steps, trajectory_count


def load_legacy_actor(args, device, model_fn):
    model = model_fn(args)

    checkpoint = './pretrained_models/LocomotionEnv/cassie-LocomotionEnv/10-27-17-03/actor.pt'

    checkpoint = torch.load(checkpoint, map_location=device)['model_state_dict']

    model.load_state_dict(checkpoint)

    return model


def delete_all_files_in_dir(dir_path):
    for file in glob.glob(os.path.join(dir_path, '*')):
        if os.path.isfile(file) or os.path.islink(file):
            os.unlink(file)
        elif os.path.isdir(file):
            shutil.rmtree(file)


def load_actor(args, device):
    if args.model_checkpoint == 'latest':
        checkpoint_path = f'checkpoints/checkpoint-{args.run_name}.pt'
    else:
        checkpoint_path = f'checkpoints/checkpoint-{args.run_name}-{args.model_checkpoint}.pt'

    config_path = os.path.join(os.path.dirname(checkpoint_path), f'config-{args.run_name}.yaml')

    module = importlib.import_module('algo.common.network')

    if not args.offline:
        wandb.login()
        run = wandb.Api().run(os.path.join(args.project_name, args.run_name.replace(':', '_')))
        run.file(name=checkpoint_path).download(replace=True, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(run.config, f)

    with open(checkpoint_path, 'rb') as r:
        checkpoint = torch.load(r, map_location=device)

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    args = argparse.Namespace(**{**config, **{k: v for k, v in vars(args).items() if v is not None}})

    logging.info(f'Checkpoint loading from: {args.run_name}')

    MODEL_CLASS = getattr(module, config['actor_name'])

    model = MODEL_CLASS(args)

    model.to(device)

    model.load_state_dict(checkpoint['actor_state_dict'])

    logging.info(
        f'Loaded checkpoint: {checkpoint.get("epoch", 0)}, {checkpoint.get("total_steps", 0), {checkpoint.get("trajectory_count", 0)} }')

    model.eval()

    if not args.offline:
        wandb.finish()

    return model


def get_batched_episodes(episodes):
    transition_keys = episodes[0].get_transition_keys()
    state_keys = episodes[0].get_state_keys()

    ep_lens = [len(episode) for episode in episodes]

    logging.debug(f'Episodes: {len(episodes)}, Buffer size: {sum(ep_lens)}')

    max_ep_len = max(ep_lens)
    max_num_cassie = max([episode['a'].size(1) for episode in episodes])

    for i, episode in enumerate(episodes):
        for k in transition_keys:

            # if k in ('a', 'v'):
            #     print(f'shape {k}', episodes[i][k].shape)

            if k == 's':
                for k in state_keys:
                    episodes[i]['s'][k] = pad(episodes[i]['s'][k],
                                              (*[0] * (episodes[i]['s'][k].dim() - 2) * 2,
                                               0, max_num_cassie - episodes[i]['s'][k].size(1),
                                               0, max_ep_len - episodes[i]['s'][k].size(0) + 1))
            elif k == 'v':
                episodes[i][k] = pad(episodes[i][k], (*[0] * (episodes[i][k].dim() - 2) * 2,
                                                      0, max_num_cassie - episodes[i][k].size(1),
                                                      0, max_ep_len - episodes[i][k].size(0) + 1))
            else:
                episodes[i][k] = pad(episodes[i][k], (*[0] * (episodes[i][k].dim() - 2) * 2,
                                                      0, max_num_cassie - episodes[i][k].size(1),
                                                      0, max_ep_len - episodes[i][k].size(0)))

    batched_episode = {'s': OrderedDict()}
    for k in transition_keys:
        if k == 's':
            for k in state_keys:
                batched_episode['s'][k] = torch.stack([episode['s'][k] for episode in episodes], dim=0)
            continue

        batched_episode[k] = torch.stack([episode[k] for episode in episodes], dim=0)

    batched_episode['ep_lens'] = torch.tensor(ep_lens)
    #
    # print(f'batched_episode s cmd shape', batched_episode['s']['cmd'].requires_grad)
    # print(f'batched_episode v shape', batched_episode['v'].requires_grad)
    # print(f'batched_episode a shape', batched_episode['a'].requires_grad)

    return batched_episode


def compute_advantages(batched_episode, args, value_function, device):
    # batched_episode: [num_episodes, seq_len, num_cassie, seq_len, *]

    E, T, N = batched_episode['a'].size()[:3]

    s = OrderedDict()
    for k in batched_episode['s'].keys():
        s[k] = batched_episode['s'][k].to(device)

    # Only compute value using value function if value does not exist in the buffer
    if 'v' not in batched_episode:
        with torch.inference_mode():
            v_ = []

            # Set the batch size if buffer does not fit in device
            batch_size = 32
            # batch_size = E
            for i in range(0, E, batch_size):
                if hasattr(value_function, 'init_hidden_state'):
                    value_function.init_hidden_state(device, batch_size=min(batch_size, E - i) * N)

                if 'Transformer' in args.critic_name:
                    src_key_padding_mask = (~batched_episode['active'][i:i + batch_size].to(device)).int()
                    # Add padding to the last sequence element to account for the last state
                    src_key_padding_mask = pad(src_key_padding_mask.float(), (0, 0, 0, 1), mode='replicate').bool()
                else:
                    src_key_padding_mask = None

                s_ = OrderedDict()
                for k in s.keys():
                    if k == 's':
                        for k in s['s'].keys():
                            s_[k] = s['s'][k][i:i + batch_size]
                        continue

                    s_[k] = s[k][i:i + batch_size]

                if 'Transformer' in args.critic_name:
                    v = value_function.forward(s_, src_key_padding_mask=src_key_padding_mask).squeeze(-1)
                else:
                    v = value_function.forward(s_).squeeze(-1)

                v_.append(v)

            if args.empty_cuda_cache and device.type == 'cuda':
                torch.cuda.empty_cache()

            v = torch.cat(v_, dim=0)
    else:
        v = batched_episode['v'].to(device)
        # print(
        #     f"using batch from memory {v.shape}, action shape {batched_episode['a'].shape}, state {batched_episode['s']['cmd'].shape}")

    v_next = v[:, 1:]
    v = v[:, :-1]

    v = v.transpose(1, 2)
    v = v.reshape(-1, *v.shape[2:])

    v_next = v_next.transpose(1, 2)
    v_next = v_next.reshape(-1, *v_next.shape[2:])

    active = batched_episode['active'].transpose(1, 2).to(device)
    active = active.reshape(-1, *active.shape[2:])

    r = batched_episode['r'].transpose(1, 2).to(device)
    r = r.reshape(-1, *r.shape[2:])

    dw = batched_episode['dw'].transpose(1, 2).to(device)
    dw = dw.reshape(-1, *dw.shape[2:])

    adv = torch.zeros_like(r, device=r.device)
    gae = 0

    # print('r', r.shape, 'v_next', v_next.shape, 'v', v.shape, 'active', active.shape)
    deltas = (r + args.gamma * v_next * ~dw - v) * active

    for t in reversed(range(r.size(1))):
        gae = deltas[:, t] + args.gamma * args.lamda * gae
        adv[:, t] = gae

    v_target = adv + v
    if args.use_adv_norm:
        mean = adv[active].mean()
        std = adv[active].std() + 1e-8
        adv = (adv - mean) / std

    adv = adv * active
    v_target = v_target * active

    adv = adv.reshape(E, N, T).transpose(1, 2)
    v_target = v_target.reshape(E, N, T).transpose(1, 2)

    return adv, v_target


def load_tarsus_predictor(args, device, model_fn):
    model = model_fn(args)
    model.to(device)

    wandb.login()

    run = wandb.Api().run(os.path.join(args.project_name, args.run_name.replace(':', '_')))

    logging.info(f'Checkpoint loading from: {args.run_name}')

    if args.model_checkpoint == 'latest':
        checkpoint_path = f'checkpoints/checkpoint-{args.run_name}.pt'

        run.file(name=checkpoint_path).download(replace=args.redownload_checkpoint, exist_ok=True)

        with open(checkpoint_path, 'rb') as r:
            checkpoint = torch.load(r, map_location=device)

        model.load_state_dict(checkpoint['state_dict'])

        logging.info(
            f'Loaded checkpoint: {checkpoint.get("epoch", 0)}, {checkpoint.get("total_steps", 0), {checkpoint.get("trajectory_count", 0)} }')
    else:
        if args.model_checkpoint == 'best':
            model_path = f'saved_models/model-{args.run_name}.pth'
        else:
            model_path = f'saved_models/model-{args.run_name}-{args.model_checkpoint}.pth'

        run.file(name=model_path).download(replace=args.redownload_checkpoint, exist_ok=True)

        with open(model_path, 'rb') as r:
            checkpoint = torch.load(r, map_location=device)

        model.load_state_dict(checkpoint)

        logging.info(f'Loaded model: {args.model_checkpoint}')

    model.eval()

    wandb.finish()

    return model


def get_device(args):
    if 'cuda' in args.device and torch.cuda.is_available():
        return torch.device('cpu'), torch.device(args.device)
    elif 'mps' in args.device and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"
        return torch.device('cpu'), torch.device(args.device)
    else:
        return torch.device('cpu'), torch.device('cpu')


def update_model(model, new_model_params):
    for p, new_p in zip(model.parameters(), new_model_params):
        p.data.copy_(new_p)


@ray.remote
class Dispatcher:
    def __init__(self):
        self.buffer_size = defaultdict(lambda: 0)

    def increment_buffer_size(self, num_cassie, size):
        self.buffer_size[num_cassie] += size

    def get_buffer_size(self, num_cassie):
        return self.buffer_size[num_cassie]

    def get_buffer_sizes(self):
        return self.buffer_size

    def reset_buffer_size(self):
        self.buffer_size = defaultdict(lambda: 0)


def get_action(actor, s, deterministic=False, src_key_padding_mask=None, return_log_prob=False):
    with torch.inference_mode():
        if deterministic:
            if src_key_padding_mask is None:
                a, _ = actor.forward(s)
            else:
                a, _ = actor.forward(s, src_key_padding_mask)
            # Get output from last observation

            return a
        else:
            if src_key_padding_mask is None:
                dist = actor.pdf(s)
            else:
                dist = actor.pdf(s, src_key_padding_mask)
            a = dist.sample()
            # a: [1, seq_len, action_dim]

            if return_log_prob:
                return a, dist.log_prob(a).sum(-1)

            return a, None


def _get_value(s, critic, src_key_padding_mask):
    with torch.inference_mode():
        if src_key_padding_mask is None:
            v = critic.forward(s).squeeze(-1)
        else:
            v = critic.forward(s, src_key_padding_mask=src_key_padding_mask).squeeze(-1)

        return v


def set_metadata_as_mtime(file_path, metadata):
    # Convert the integer metadata to a timestamp (in seconds since the epoch)
    # You can choose an epoch time (like 1970-01-01) and add the metadata as an offset
    epoch_time = 1609459200  # January 1, 2021, 00:00:00 (example epoch time)
    new_mtime = epoch_time + metadata

    # Set both access and modified times to the new timestamp
    os.utime(file_path, (new_mtime, new_mtime))


def get_metadata_from_mtime(file_path):
    # Retrieve the modified time of the file
    mtime = os.path.getmtime(file_path)

    # Convert the modified time back to the integer metadata
    epoch_time = 1609459200  # Same epoch time used in set_metadata_as_mtime
    metadata = int(mtime - epoch_time)

    return metadata


def extract_episode_iteration(filename):
    match = re.search(r'episode_(\d+)_\d+_\d+.xz', filename)
    return int(match.group(1))


def read_episode_file(episode_file):
    try:
        with lzma.open(episode_file, 'rb') as f:
            episode = pickle.load(f)
        return episode
    except EOFError as e:
        # logging.error(f'Error loading episode file: {episode_file}.')
        # logging.error(e, exc_info=True)
        pass


@ray.remote
class Worker:
    def __init__(self, env, actor, args, device, worker_id, dispatcher=None, critic=None):
        self.dispatcher = dispatcher

        # Normalize probability by number of cassies
        num_cassie_prob_norm = args.num_cassie_prob * np.arange(1, len(args.num_cassie_prob) + 1)
        num_cassie_prob_norm /= num_cassie_prob_norm.sum()

        num_cassie = np.random.choice(np.arange(1, len(num_cassie_prob_norm) + 1), p=num_cassie_prob_norm)

        self.env = env(num_cassie)

        self.env.eval(False)

        # self.dispatcher = dispatcher
        if args.use_reward_scaling:
            self.reward_scaling = RewardScaling(shape=1, gamma=args.gamma)
        self.args = args
        self.device = device
        self.actor = deepcopy(actor).to(device)
        self.actor.eval()

        if critic:
            self.critic = deepcopy(critic).to(device)
            self.critic.eval()

        self.worker_id = worker_id

        if self.env._merge_states:
            self.num_cassie = 1
        else:
            self.num_cassie = self.env.num_cassie

    def get_num_cassie(self):
        return self.num_cassie

    def get_buffer_size(self):
        return self.args.time_horizon

    def update_actor(self, new_actor_params):
        for p, new_p in zip(self.actor.parameters(), new_actor_params):
            p.data.copy_(new_p)

    def update_critic(self, new_actor_params):
        for p, new_p in zip(self.critic.parameters(), new_actor_params):
            p.data.copy_(new_p)

    def save_episode(self, data_path, episode):
        with open(data_path, 'wb') as f:
            pickle.dump(episode, f)

    def sample_episode(self, ep_len, store_log_prob=False):
        torch.set_num_threads(1)

        store_value = hasattr(self, 'critic')

        episode = Episode(ep_len, self.num_cassie, self.env.observation_size,
                          self.env.action_size, store_log_prob=store_log_prob, store_value=store_value)

        episode_reward = np.zeros(self.num_cassie)

        s = self.env.reset()

        if hasattr(self.actor, 'init_hidden_state'):
            self.actor.init_hidden_state(self.device, batch_size=self.num_cassie)

        if store_value and hasattr(self.critic, 'init_hidden_state'):
            self.critic.init_hidden_state(self.device, batch_size=self.num_cassie)

        if self.args.use_reward_scaling:
            self.reward_scaling.reset()

        active = torch.ones(self.num_cassie, dtype=torch.bool, device=self.device)

        if 'Transformer' in self.args.actor_name:
            src_key_padding_mask = torch.zeros(1, 1, self.num_cassie, dtype=torch.bool)
        else:
            src_key_padding_mask = None

        for step in range(ep_len):
            # Numpy to tensor

            s_ = OrderedDict()

            for k in s.keys():
                # Add batch and sequence dimension
                s_[k] = torch.tensor(s[k], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                # s[k]: [batch(E)=1, seq_len(T)=1, num_cassie(N), *F]

            arg = get_action(self.actor, s_,
                             deterministic=False,
                             src_key_padding_mask=src_key_padding_mask,
                             return_log_prob=store_log_prob)

            if store_value:
                v = _get_value(s_, self.critic, src_key_padding_mask)

            a = arg[0].squeeze(0).squeeze(0)

            if store_log_prob:
                log_prob = arg[1]

            s_next, r, done, _ = self.env.step(a.numpy())
            done = torch.tensor(done, dtype=torch.bool, device=self.device)

            # episode.store_transition(kv=dict(active=~done))

            episode_reward += r

            # True if episode ended before time horizon, False otherwise
            dw = done & (step != self.args.time_horizon - 1)

            if self.args.use_reward_scaling:
                r = self.reward_scaling(r)

            r = torch.tensor(r, dtype=torch.float32, device=self.device) * active

            # Remove batch and sequence dimension
            for k in s.keys():
                s_[k] = s_[k].squeeze(0).squeeze(0)

            T = dict(s=s_, a=a, r=r, dw=dw, active=active)

            if store_log_prob:
                T['log_prob'] = log_prob

            if store_value:
                T['v'] = v

            episode.store_transition(kv=T)
            episode.step()

            active &= ~done

            s = s_next

            if done.all():
                break

        s_ = OrderedDict()
        for k in s.keys():
            s_[k] = torch.tensor(s[k], dtype=torch.float32)

        T = dict(s=s_)

        if store_value:
            s__ = OrderedDict()
            for k in s.keys():
                s__[k] = s_[k].unsqueeze(0).unsqueeze(0)

            T['v'] = _get_value(s__, self.critic, src_key_padding_mask)

        episode.store_transition(kv=T)

        # Truncate the episode to the actual length
        episode.end()

        return episode, episode_reward, step + 1, self.worker_id, self.num_cassie

    def dump_episodes(self, iteration, target_buffer_size):
        curr_buffer_size = 0

        for episode_id in count():
            rem_buffer_size = target_buffer_size - curr_buffer_size

            if rem_buffer_size <= 0:
                break

            ep_len = min(self.args.time_horizon, rem_buffer_size)

            # Get only the first argument which is the episode
            episode = self.sample_episode(ep_len, store_log_prob=True)[0]

            episode.meta_data = {'iteration': iteration}

            data_path = os.path.join(self.args.sample_local_dir, f'worker_{self.worker_id}_{episode_id}.pkl')

            with open(data_path, 'wb') as f:
                pickle.dump(episode, f)

            curr_buffer_size += len(episode)

    def dump_episodes_v2(self):
        episode_id = 0
        prev_buffer_size = 0
        rem_buffer_size = self.args.buffer_size
        # total_steps = 0

        # tq = tqdm.tqdm(desc='Total steps', position=0, colour='cyan')

        for _ in count():
            self.actor, self.critic, iteration, buffer_size = ray.get([self.dispatcher.get_actor.remote(),
                                                                       self.dispatcher.get_critic.remote(),
                                                                       self.dispatcher.get_iteration.remote(),
                                                                       self.dispatcher.get_buffer_size.remote(
                                                                           self.num_cassie)])

            episode = self.sample_episode(self.args.time_horizon, store_log_prob=True)[0]

            episode.meta_data = {'iteration': iteration}

            # curr_buffer_size += len(episode)

            self.dispatcher.increment_buffer_size.remote(self.num_cassie, len(episode))

            # total_steps += len(episode)
            if buffer_size - prev_buffer_size > self.args.buffer_size:
                prev_buffer_size = buffer_size
                episode_id = 0
            else:
                episode_id += 1

            # data_path = os.path.join(self.args.sample_local_dir,
            #                          f'episode_{iteration % self.args.iteration_TOL}_{self.worker_id}_{episode_id}.xz')

            data_path = os.path.join(self.args.sample_local_dir,
                                     f'episode_{self.worker_id}_{episode_id}.xz')

            logging.info(
                f'Itr: {iteration} | Worker {self.worker_id} | Episode id: {episode_id} | Buffer size: {buffer_size} | Buffer size diff: {buffer_size - prev_buffer_size} | Episode size: {len(episode)}')

            # tq.set_description(
            #     f'Itr: {iteration}')
            # tq.update(len(episode))

            with lzma.open(data_path, 'wb') as f:
                pickle.dump(episode, f)

            set_metadata_as_mtime(data_path, iteration)

            # print('meta data', get_metadata_from_mtime(data_path))

    def evaluate(self, max_ep_len):
        torch.set_num_threads(1)

        with torch.inference_mode():
            s = self.env.reset()

            if hasattr(self.actor, 'init_hidden_state'):
                self.actor.init_hidden_state(self.device, batch_size=self.num_cassie)

            episode_reward = 0

            active = np.ones(self.num_cassie, dtype=bool)
            src_key_padding_mask = torch.zeros(1, self.num_cassie, dtype=torch.bool)

            for step in range(max_ep_len):
                # Numpy to tensor
                s_ = OrderedDict()
                for k in s.keys():
                    s_[k] = torch.tensor(s[k], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

                a = self.get_action(s_,
                                    src_key_padding_mask=src_key_padding_mask,
                                    deterministic=True).squeeze(0).squeeze(0)

                s, r, done, _ = self.env.step(a.numpy())

                episode_reward += r * active

                active &= ~done

                if done.all():
                    break

            return None, episode_reward, step + 1, self.worker_id, self.num_cassie


import numpy as np


class BatchedPerlinNoise:

    # Fade function (smoothing function)
    def fade(self, t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    # Linear interpolation function
    def lerp(self, t, a, b):
        return a + t * (b - a)

    # Generate a grid of gradient vectors
    def generate_gradients(self, shape):
        angles = np.random.rand(*shape) * 2 * np.pi
        gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
        return gradients

    # Perlin noise function
    def perlin_noise_2d(self, grid_size, tile_size):
        # Create the grid of points
        grid_x, grid_y = np.meshgrid(np.arange(grid_size[0]), np.arange(grid_size[1]), indexing='ij')

        # Lattice points
        lattice_x = grid_x // tile_size
        lattice_y = grid_y // tile_size

        # Gradient vectors at lattice points
        gradients = self.generate_gradients((grid_size[0] // tile_size + 1, grid_size[1] // tile_size + 1))

        # Compute vectors from lattice points to each grid point
        dx = (grid_x % tile_size) / tile_size
        dy = (grid_y % tile_size) / tile_size

        # Compute dot products between gradient vectors and distance vectors
        g00 = gradients[lattice_x, lattice_y]
        g10 = gradients[lattice_x + 1, lattice_y]
        g01 = gradients[lattice_x, lattice_y + 1]
        g11 = gradients[lattice_x + 1, lattice_y + 1]

        # Dot products
        dot00 = g00[..., 0] * dx + g00[..., 1] * dy
        dot10 = g10[..., 0] * (dx - 1) + g10[..., 1] * dy
        dot01 = g01[..., 0] * dx + g01[..., 1] * (dy - 1)
        dot11 = g11[..., 0] * (dx - 1) + g11[..., 1] * (dy - 1)

        # Fade curves
        u = self.fade(dx)
        v = self.fade(dy)

        # Interpolate along x
        lerp_x0 = self.lerp(u, dot00, dot10)
        lerp_x1 = self.lerp(u, dot01, dot11)

        # Interpolate along y
        perlin = self.lerp(v, lerp_x0, lerp_x1)

        # Calculate distance from centroid
        centroid_x, centroid_y = grid_size[0] / 2, grid_size[1] / 2
        distance_from_centroid = np.sqrt((grid_x - centroid_x) ** 2 + (grid_y - centroid_y) ** 2)

        # Apply a weighting function to flatten near the centroid
        max_distance = np.sqrt((grid_size[0] / 2) ** 2 + (grid_size[1] / 2) ** 2)
        weight = 1 - np.exp(-3 * (distance_from_centroid / max_distance) ** 2)

        # Apply the weight to the Perlin noise
        perlin *= weight

        return perlin


class BatchedPerlinNoise2:

    def fade(self, t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    # Linear interpolation function
    def lerp(self, t, a, b):
        return a + t * (b - a)

    # Generate a grid of gradient vectors
    def generate_gradients(self, shape, seed):
        np.random.seed(seed)
        angles = np.random.rand(*shape) * 2 * np.pi
        gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
        return gradients

    # Single octave of Perlin noise
    def perlin_noise_2d(self, grid_size, tile_size, seed):
        grid_x, grid_y = np.meshgrid(np.arange(grid_size[0]), np.arange(grid_size[1]), indexing='ij')

        lattice_x = grid_x // tile_size
        lattice_y = grid_y // tile_size

        gradients = self.generate_gradients((grid_size[0] // tile_size + 1, grid_size[1] // tile_size + 1), seed)

        dx = (grid_x % tile_size) / tile_size
        dy = (grid_y % tile_size) / tile_size

        g00 = gradients[lattice_x, lattice_y]
        g10 = gradients[lattice_x + 1, lattice_y]
        g01 = gradients[lattice_x, lattice_y + 1]
        g11 = gradients[lattice_x + 1, lattice_y + 1]

        dot00 = g00[..., 0] * dx + g00[..., 1] * dy
        dot10 = g10[..., 0] * (dx - 1) + g10[..., 1] * dy
        dot01 = g01[..., 0] * dx + g01[..., 1] * (dy - 1)
        dot11 = g11[..., 0] * (dx - 1) + g11[..., 1] * (dy - 1)

        u = self.fade(dx)
        v = self.fade(dy)

        lerp_x0 = self.lerp(u, dot00, dot10)
        lerp_x1 = self.lerp(u, dot01, dot11)

        perlin = self.lerp(v, lerp_x0, lerp_x1)

        return perlin

    # Combine multiple octaves of Perlin noise
    def perlin_noise_with_octaves(self, grid_size, octaves, seed):
        noise = np.zeros(grid_size)
        frequency = 1.0
        amplitude = 1.0
        max_amplitude = 0.0

        for _ in range(octaves):
            tile_size = int(grid_size[0] // (frequency * 10))
            if tile_size == 0:
                break
            noise += amplitude * self.perlin_noise_2d(grid_size, tile_size, seed)

            max_amplitude += amplitude
            amplitude /= 2.0
            frequency *= 2.0

        # Normalize the result
        noise /= max_amplitude

        return noise


class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain
        self.previous_error = 0  # Previous error value
        self.integral = 0  # Integral of the error

    def update(self, error, delta_time):
        # Proportional term
        proportional = self.kp * error

        # Integral term
        self.integral += error * delta_time
        integral = self.ki * self.integral

        # Derivative term
        derivative = self.kd * (error - self.previous_error) / delta_time
        self.previous_error = error

        # PID output
        output = proportional + integral + derivative
        return output

    def reset(self):
        self.previous_error = 0
        self.integral = 0
