import copy

import torch.nn.functional as F
import tqdm
from torch.distributions import kl_divergence
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from algo.common.utils import *
from util.mirror import mirror_tensor


class PPO_algo:
    def __init__(self, args, device, mirror_dict):
        self.args = args
        self.device = device
        self.mirror_dict = mirror_dict

        self.num_threads = torch.get_num_threads()
        #
        # self.actor = Actor_FF_v5(args)
        # self.critic = Critic_FF_v5(args)

        module = importlib.import_module('algo.common.network')

        self.actor = getattr(module, args.actor_name)(args)
        self.critic = getattr(module, args.critic_name)(args)

        self.actor = self.actor.to(self.device)
        self.critic = self.critic.to(self.device)

        if self.args.set_adam_eps:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.args.lr_a, eps=self.args.eps)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.args.lr_c, eps=self.args.eps)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.args.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.args.lr_c)

        self.actor_old = copy.deepcopy(self.actor)
        self.actor_old.eval()

        self.pbar_epoch = tqdm.tqdm(total=args.num_epoch, desc='Training epoch',
                                    position=np.count_nonzero(self.args.num_cassie_prob) + 2, colour='green')

    def _get_mirror_loss(self, s, active, dist_now, src_key_padding_mask=None):
        s_mirrored = {}
        for k in s.keys():
            if k == 'depth':
                s_mirrored[k] = s[k].flip(dims=[-1])
                continue
            s_mirrored[k] = mirror_tensor(s[k], self.mirror_dict['state_mirror_indices'][k])
        with torch.no_grad():
            if 'Transformer' in self.args.actor_name:
                assert src_key_padding_mask is not None, 'src_key_padding_mask must be provided for Transformer'
                mirrored_a, _ = self.actor.forward(s_mirrored, src_key_padding_mask=src_key_padding_mask)
            else:
                mirrored_a, _ = self.actor.forward(s_mirrored)
        target_a = mirror_tensor(mirrored_a, self.mirror_dict['action_mirror_indices'])[active]
        mirror_loss = (0.5 * F.mse_loss(dist_now.mean[active], target_a))

        return mirror_loss

    def _get_ppo_loss(self, s, a, adv, v_target, a_logprob, active, src_key_padding_mask=None):
        # Forward pass:
        if 'Transformer' in self.args.actor_name:
            assert src_key_padding_mask is not None, 'src_key_padding_mask must be provided for Transformer'
            dist_now = self.actor.pdf(s, src_key_padding_mask=src_key_padding_mask)
        else:
            dist_now = self.actor.pdf(s)

        if 'Transformer' in self.args.critic_name:
            assert src_key_padding_mask is not None, 'src_key_padding_mask must be provided for Transformer'
            values_now = self.critic.forward(s, src_key_padding_mask=src_key_padding_mask).squeeze(-1)[active]
        else:
            values_now = self.critic.forward(s).squeeze(-1)[active]

        ratios = (dist_now.log_prob(a).sum(-1)[active] - a_logprob[active]).exp()

        # actor loss
        surr1 = ratios * adv
        surr2 = torch.clamp(ratios, 1 - self.args.epsilon, 1 + self.args.epsilon) * adv
        entropy_loss = - self.args.entropy_coef * dist_now.entropy().sum(-1)[active]
        actor_loss = -torch.min(surr1, surr2)

        actor_loss = actor_loss.mean()
        entropy_loss = entropy_loss.mean()
        critic_loss = (0.5 * F.mse_loss(values_now, v_target))

        return actor_loss, entropy_loss, critic_loss, dist_now

    def update(self, batched_episode, total_steps, check_kl, callback=None):
        torch.set_num_threads(self.num_threads)
        # batched_episode's shape: E, T, N, D

        self.pbar_epoch.reset()

        losses = []
        kls = []

        # Compute advantage and target value
        adv, v_target = compute_advantages(batched_episode, self.args, self.critic, device=self.device)

        batched_episode['adv'] = adv
        batched_episode['v_target'] = v_target

        for k in batched_episode.keys():
            if k == 's':
                for k in batched_episode['s'].keys():
                    # omit the last state
                    batched_episode['s'][k] = batched_episode['s'][k][:, :-1].to(self.device)
                continue

            batched_episode[k] = batched_episode[k].to(self.device)

        num_batches = batched_episode['a'].size(0)

        # print('batched_episode logprob shape', batched_episode['log_prob'].shape, 'batched actions', batched_episode['a'].shape)
        if 'log_prob' not in batched_episode:
            # If log_prob is provided, no need of old actor (won't be computing KL though)
            self.actor_old.load_state_dict(self.actor.state_dict())

        for i in range(self.args.num_epoch):
            self.pbar_epoch.update(1)

            early_stop = False
            sampler = BatchSampler(SubsetRandomSampler(range(num_batches)), self.args.mini_batch_size, False)
            for index in sampler:
                N = batched_episode['a'].size(2)

                if hasattr(self.actor, 'init_hidden_state'):
                    self.actor.init_hidden_state(device=self.device, batch_size=len(index) * N)

                if hasattr(self.critic, 'init_hidden_state'):
                    self.critic.init_hidden_state(device=self.device, batch_size=len(index) * N)

                if hasattr(self.actor_old, 'init_hidden_state'):
                    self.actor_old.init_hidden_state(device=self.device, batch_size=len(index) * N)

                s = OrderedDict()
                for k in batched_episode['s'].keys():
                    s[k] = batched_episode['s'][k][index]

                a = batched_episode['a'][index]

                active = batched_episode['active'][index]
                src_key_padding_mask = ~active

                adv = batched_episode['adv'][index][active]
                v_target = batched_episode['v_target'][index][active]

                # else:
                if 'log_prob' not in batched_episode:
                    # Compute action log probabilities and PDF using the rollout policy
                    with torch.inference_mode():
                        if 'Transformer' in self.args.actor_name:
                            dist_old = self.actor_old.pdf(s, src_key_padding_mask=src_key_padding_mask)
                        else:
                            dist_old = self.actor_old.pdf(s)
                        a_logprob = dist_old.log_prob(a).sum(-1)

                else:
                    a_logprob = batched_episode['log_prob'][index]

                # Get losses and new PDF
                actor_loss1, entropy_loss1, critic_loss1, dist_now = \
                    self._get_ppo_loss(s, a, adv, v_target, a_logprob, active, src_key_padding_mask)

                if 'log_prob' not in batched_episode:
                    # Compute KL between old and new PDF
                    with torch.inference_mode():
                        kl = kl_divergence(dist_now, dist_old).sum(-1)[active].mean()
                        kls.append(kl.item())

                    # Stop if new policy is too different than old
                    if self.args.kl_check and check_kl and kl > self.args.kl_threshold:
                        logging.warning(f'Early stopping at epoch {i} due to reaching max kl. kl={kl}')
                        early_stop = True
                        break

                # Supervised mirror loss
                if 'supervised' in self.args.mirror_loss:
                    if 'Transformer' in self.args.actor_name:
                        mirror_loss1 = self._get_mirror_loss(s, active, dist_now, src_key_padding_mask)
                    else:
                        mirror_loss1 = self._get_mirror_loss(s, active, dist_now)

                    mirror_loss = mirror_loss1

                if 'ppo' in self.args.mirror_loss:
                    for k in s.keys():
                        if k == 'depth':
                            s[k] = s[k].flip(dims=[-1])
                            continue
                        s[k] = mirror_tensor(s[k], self.mirror_dict['state_mirror_indices'][k])

                    a = mirror_tensor(a, self.mirror_dict['action_mirror_indices'])

                    # Compute mirrored action log probabilities and PDF using the rollout policy
                    actor_loss2, entropy_loss2, critic_loss2, dist_now = \
                        self._get_ppo_loss(s, a, adv, v_target, a_logprob, active, src_key_padding_mask)

                    # Supervised mirror loss on top of PPO mirror loss. Doubly mirrored
                    if 'supervised' in self.args.mirror_loss:
                        if 'Transformer' in self.args.actor_name:
                            mirror_loss2 = self._get_mirror_loss(s, active, dist_now, src_key_padding_mask)
                        else:
                            mirror_loss2 = self._get_mirror_loss(s, active, dist_now)

                        mirror_loss = (mirror_loss1 + mirror_loss2) / 2.

                    actor_loss = (actor_loss1 + actor_loss2) / 2.
                    entropy_loss = (entropy_loss1 + entropy_loss2) / 2.
                    critic_loss = (critic_loss1 + critic_loss2) / 2.
                else:
                    actor_loss = actor_loss1
                    entropy_loss = entropy_loss1
                    critic_loss = critic_loss1

                log = {'epochs': i, 'actor_loss': actor_loss.item(), 'entropy_loss': entropy_loss.item(),
                       'critic_loss': critic_loss.item(), 'num_batches': num_batches,
                       'active_count': active.sum().item(), 'active_shape': active.shape}

                if 'log_prob' not in batched_episode:
                    log['kl_divergence'] = kl

                if 'supervised' in self.args.mirror_loss:
                    log['mirror_loss'] = mirror_loss.item()
                    losses.append((log['actor_loss'], log['entropy_loss'], log['mirror_loss'], log['critic_loss']))
                else:
                    losses.append((log['actor_loss'], log['entropy_loss'], log['critic_loss']))

                self.optimizer_actor.zero_grad()
                self.optimizer_critic.zero_grad()

                if 'supervised' in self.args.mirror_loss:
                    (actor_loss + entropy_loss + mirror_loss).backward()
                else:
                    (actor_loss + entropy_loss).backward()
                critic_loss.backward()

                if self.args.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.args.grad_clip)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.args.grad_clip)

                if any([((~param.grad.isfinite()).any()).item() for param in self.actor.parameters() if
                        param.grad is not None]):

                    # collect gradients in array
                    gradients = []
                    for param in self.actor.parameters():
                        if param.grad is None:
                            continue
                        gradients.append(param.grad)

                    torch.save({'s': s, 'a': a, 'adv': adv, 'v_target': v_target,
                                'actor_old_state_dict': self.actor_old.state_dict(),
                                'actor_state_dict': self.actor.state_dict(),
                                'optimizer_actor_state_dict': self.optimizer_actor.state_dict(),
                                'gradients': gradients,
                                'entropy_loss': entropy_loss, 'actor_loss': actor_loss,
                                'critic_loss': critic_loss, 'active_seq': active},
                               f'training_logs/training_error_{self.args.run_name}.pt')
                    # raise RuntimeError(
                    #     f"Non-finite values detected in gradients. Saved to training_logs/training_error_{self.args.run_name}.pt'")
                    logging.warning(
                        f"Non-finite values detected in gradients. Saved to training_logs/training_error_{self.args.run_name}.pt'")
                    early_stop = True
                    break

                self.optimizer_actor.step()
                self.optimizer_critic.step()

                del s, adv, actor_loss, entropy_loss, critic_loss, v_target, active, dist_now,

                if self.args.empty_cuda_cache and self.device.type == 'cuda':
                    torch.cuda.empty_cache()

                if callback is not None:
                    callback()

            if early_stop:
                break

        if self.args.use_lr_decay:
            self.lr_decay(total_steps)

        if 'supervised' in self.args.mirror_loss:
            a_loss, e_loss, m_loss, c_loss = zip(*losses)
            a_loss, e_loss, m_loss, c_loss = np.mean(a_loss), np.mean(e_loss), np.mean(m_loss), np.mean(c_loss)
        else:
            a_loss, e_loss, c_loss = zip(*losses)
            a_loss, e_loss, c_loss = np.mean(a_loss), np.mean(e_loss), np.mean(c_loss)
            m_loss = None

        if len(kls) > 0:
            kl = np.mean(kls)
        else:
            kl = 0

        del losses, kls, batched_episode

        if self.args.empty_cuda_cache and self.device.type == 'cuda':
            torch.cuda.empty_cache()

        self.pbar_epoch.reset()

        return a_loss, e_loss, m_loss, c_loss, kl, num_batches, i

    def lr_decay(self, total_steps):
        lr_a_now = self.args.lr_a * (1 - total_steps / self.args.max_steps)
        lr_c_now = self.args.lr_c * (1 - total_steps / self.args.max_steps)
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now
