import argparse
import datetime
from collections import defaultdict

import tqdm

from algo.common.ppo_algo import PPO_algo
from algo.common.utils import *
from env.cassie.cassiepede.cassiepede import Cassiepede


def main():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Normalize the probability of number of cassie
    args.num_cassie_prob = np.array(args.num_cassie_prob, dtype=float) / np.sum(args.num_cassie_prob)

    dev_cpu, dev_gpu = get_device(args)

    args.device = dev_gpu.__str__()

    custom_terrain = None

    env_fn = lambda num_cassie: Cassiepede(
        reward_name=args.reward_name,
        simulator_type='mujoco',
        policy_rate=50,
        dynamics_randomization=True,
        state_noise=[0.05, 0.1, 0.01, 0.2, 0.01, 0.2, 0.05, 0.0, 0.0],
        velocity_noise=0.0,
        state_est=False,
        integral_action=False,
        depth_input=False,
        num_cassie=num_cassie,
        custom_terrain=custom_terrain,
        only_deck_force=False,
        height_control=True,
        merge_states=False,
        state_history_size=args.state_history_size,
        poi_position_offset=0.0 if num_cassie == 1 else args.poi_position_offset,
        perturbation_force=args.perturbation_force,
        perturbation_torque=args.perturbation_torque,
        force_prob=args.force_prob,
        position_offset=0.0 if num_cassie == 1 else args.position_offset,
        poi_heading_range=0.0 if num_cassie == 1 else args.poi_heading_range,
        cmd_noise=args.cmd_noise,
        cmd_noise_prob=args.cmd_noise_prob,
        mask_tarsus_input=args.mask_tarsus_input,
        enable_hfield=False,
        offscreen=False)

    env = env_fn(num_cassie=np.nonzero(args.num_cassie_prob)[0][0] + 1)

    """Training code"""
    max_reward = float('-inf')
    time_now = datetime.datetime.now()

    args.env_name = env.__class__.__name__
    args.state_dim = env.observation_size
    args.action_dim = env.action_size
    args.reward_name = env.reward_name

    args.run_name = str(time_now)

    if args.mirror_loss:
        mirror_dict = env.get_mirror_dict()

        for k in mirror_dict['state_mirror_indices'].keys():
            mirror_dict['state_mirror_indices'][k] = torch.tensor(mirror_dict['state_mirror_indices'][k],
                                                                  dtype=torch.float32,
                                                                  device=dev_gpu)

        mirror_dict['action_mirror_indices'] = torch.tensor(mirror_dict['action_mirror_indices'],
                                                            dtype=torch.float32,
                                                            device=dev_gpu)
    else:
        mirror_dict = None

    agent = PPO_algo(args, device=dev_gpu, mirror_dict=mirror_dict)
    #
    # args.actor_name = agent.actor.__class__.__name__
    # args.critic_name = agent.critic.__class__.__name__

    args.num_param_actor = sum(p.numel() for p in agent.actor.parameters())
    args.num_param_critic = sum(p.numel() for p in agent.critic.parameters())

    logging.info(args)

    logging.info(f'Using device:{dev_cpu}(inference), {dev_gpu}(optimization)\n')

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('training_logs', exist_ok=True)

    run, iterations, total_steps, trajectory_count = init_logger(args, agent)

    prev_save_steps = 0

    # wandb.watch(models=agent.actor, log_freq=1)

    actor_global = deepcopy(agent.actor).to(dev_cpu)
    critic_global = deepcopy(agent.critic).to(dev_cpu)

    logging.info(f'Parameters, actor: {args.num_param_actor}, critic: {args.num_param_critic}')

    collectors = [Worker.remote(env_fn, actor_global, args, dev_cpu, i) for i in range(args.n_collectors)]

    evaluators = [Worker.remote(env_fn, actor_global, args, dev_cpu, i) for i in range(args.n_evaluators)]

    # Store worker id for each num_cassie worker. This is useful to cancel all workers for a particular num_cassie
    num_cassie_worker_map = defaultdict(lambda: [])
    for wid, num_cassie in enumerate(ray.get([collector.get_num_cassie.remote() for collector in collectors])):
        num_cassie_worker_map[num_cassie].append(wid)

    # Initializes replay buffer for each type of environment (number of cassie in it)
    target_buffer_sizes = {}
    for num_cassie in np.nonzero(args.num_cassie_prob)[0]:
        target_buffer_sizes[num_cassie + 1] = int(np.ceil(args.buffer_size * args.num_cassie_prob[num_cassie]))

    pbar_total_steps = tqdm.tqdm(total=args.max_steps, desc='Total steps', position=0, colour='cyan')

    pbar_evaluator = tqdm.tqdm(total=args.eval_buffer_size, desc='Evaluating', position=1, colour='yellow')

    pbar_collector = {
        num_cassie: tqdm.tqdm(total=target_buffer_sizes[num_cassie],
                              desc=f'Collecting [num_cassie={num_cassie},workers={len(num_cassie_worker_map[num_cassie])}]',
                              position=i + 2,
                              colour='blue')
        for i, num_cassie in enumerate(target_buffer_sizes.keys())
    }

    while total_steps < args.max_steps:
        actor_param_id = ray.put(list(actor_global.parameters()))

        evaluator_ids = {}
        if args.n_evaluators and iterations > 0 and iterations % args.evaluate_freq == 0:
            """Evaluation"""
            # logging.debug("Evaluating")
            time_evaluating = datetime.datetime.now()

            # Copy the latest actor to all evaluators
            for evaluator in evaluators:
                evaluator.update_actor.remote(actor_param_id)


            # Start the evaluators
            evaluator_ids = {
                i: evaluator.evaluate.remote(max_ep_len=min(args.time_horizon, args.eval_buffer_size))
                for i, evaluator in enumerate(evaluators)}

        """Collect data"""
        # logging.debug("Collecting")
        time_collecting = datetime.datetime.now()

        # Copy the latest actor to all collectors
        for collector in collectors:
            collector.update_actor.remote(actor_param_id)

        # Start the collectors
        collector_ids = {i: collector.sample_episode.remote(ep_len=min(args.time_horizon, args.buffer_size))
                         for i, collector in enumerate(collectors)}

        evaluator_steps = 0
        eval_rewards = defaultdict(lambda: [])
        eval_lengths = defaultdict(lambda: [])

        train_rewards = defaultdict(lambda: [])
        train_lengths = defaultdict(lambda: [])

        episodes = []
        curr_buffer_sizes = defaultdict(lambda: 0)

        # Reset the progress bar
        for pc in pbar_collector.values():
            pc.reset()
        pbar_evaluator.reset()

        while evaluator_ids or collector_ids:
            done_ids, remain_ids = ray.wait(list(collector_ids.values()) + list(evaluator_ids.values()), num_returns=1)

            episode, episode_reward, episode_length, worker_id, num_cassie = ray.get(done_ids)[0]

            if episode is None:
                # This worker is evaluator
                eval_rewards[num_cassie].append(episode_reward)
                eval_lengths[num_cassie].append(episode_length)

                evaluator_steps += episode_length

                rem_buffer_size = args.eval_buffer_size - evaluator_steps

                # Update the progress bar
                pbar_evaluator.n = min(pbar_evaluator.total, evaluator_steps)
                pbar_evaluator.refresh()

                if rem_buffer_size > 0:
                    logging.debug(f"{rem_buffer_size} steps remaining to evaluate")
                    evaluator_ids[worker_id] = evaluators[worker_id].evaluate.remote(
                        max_ep_len=min(args.time_horizon, rem_buffer_size))
                else:
                    time_evaluating = datetime.datetime.now() - time_evaluating
                    logging.debug('Evaluation done. Cancelling stale evaluators')
                    # ray.get(dispatcher.set_evaluating.remote(False))
                    map(ray.cancel, evaluator_ids.values())
                    evaluator_ids.clear()
            else:
                # This worker is collector
                train_rewards[num_cassie].append(episode_reward)
                train_lengths[num_cassie].append(episode_length)

                episodes.append(episode)
                curr_buffer_sizes[num_cassie] += len(episode)

                # Update the progress bar
                pbar_collector[num_cassie].n = min(target_buffer_sizes[num_cassie], curr_buffer_sizes[num_cassie])
                pbar_collector[num_cassie].refresh()

                if curr_buffer_sizes[num_cassie] < target_buffer_sizes[num_cassie]:
                    logging.debug(
                        f"{target_buffer_sizes[num_cassie] - curr_buffer_sizes[num_cassie]} steps remaining to collect [num_cassie={num_cassie}]")
                    collector_ids[worker_id] = collectors[worker_id].sample_episode.remote(
                        ep_len=min(args.time_horizon, target_buffer_sizes[num_cassie] - curr_buffer_sizes[num_cassie]))
                else:
                    # Prevent collecting for this num_cassie
                    for worker_id in num_cassie_worker_map[num_cassie]:
                        ray.cancel(collector_ids[worker_id])
                        del collector_ids[worker_id]
                        logging.debug(f'Collector done [num_cassie={num_cassie}]. Removing from collectors worker')

                    if len(collector_ids) == 0:
                        time_collecting = datetime.datetime.now() - time_collecting

        [ray.cancel(c) for c in list(collector_ids.values()) + list(evaluator_ids.values())]

        if args.n_evaluators and iterations > 0 and iterations % args.evaluate_freq == 0:
            eval_rewards = dict([(f'eval/episode_reward/num_cassie_{k}', np.mean(v)) for k, v in eval_rewards.items()])
            eval_lengths = dict([(f'eval/episode_length/num_cassie_{k}', np.mean(v)) for k, v in eval_lengths.items()])

            reward = np.mean(list(eval_rewards.values()))
            length = np.mean(list(eval_lengths.values()))

            log = {'eval/episode_reward': reward,
                   'eval/episode_length': length,
                   **eval_rewards,
                   **eval_lengths,
                   'misc/total_steps': total_steps,
                   'misc/iterations': iterations,
                   'misc/time_evaluating': time_evaluating.total_seconds(),
                   'misc/evaluation_rate': evaluator_steps / time_evaluating.total_seconds(),
                   'misc/time_elapsed': (datetime.datetime.now() - time_now).total_seconds()}

            logging.debug(log)
            run.log(log, step=total_steps)

        train_rewards = dict([(f'train/episode_reward/num_cassie_{k}', np.mean(v)) for k, v in train_rewards.items()])
        train_lengths = dict([(f'train/episode_length/num_cassie_{k}', np.mean(v)) for k, v in train_lengths.items()])

        mean_train_rewards = np.mean(list(train_rewards.values()))
        mean_train_lens = np.mean(list(train_lengths.values()))

        buffer_size = sum(curr_buffer_sizes.values())
        total_steps += buffer_size
        trajectory_count += len(episodes)

        log = {'train/episode_reward': mean_train_rewards,
               'train/episode_length': mean_train_lens,
               **train_rewards,
               **train_lengths,
               'misc/trajectory_count': trajectory_count,
               'misc/current_steps': buffer_size,
               'misc/total_steps': total_steps,
               'misc/collection_rate': buffer_size / time_collecting.total_seconds(),
               'misc/iterations': iterations,
               'misc/total_steps_rate': total_steps / (datetime.datetime.now() - time_now).total_seconds(),
               'misc/time_collecting': time_collecting.total_seconds(),
               'misc/time_elapsed': (datetime.datetime.now() - time_now).total_seconds()}

        logging.debug(log)
        run.log(log, step=total_steps)

        """Training"""
        logging.debug("Training")
        time_training = datetime.datetime.now()

        batch = get_batched_episodes(episodes)

        actor_loss, entropy_loss, mirror_loss, critic_loss, kl, num_batches, train_epoch = \
            agent.update(batch, total_steps, check_kl=args.kl_check_min_itr <= iterations)

        pbar_total_steps.update(buffer_size)

        # Copy updated models to global models
        update_model(actor_global, agent.actor.parameters())
        update_model(critic_global, agent.critic.parameters())

        time_training = datetime.datetime.now() - time_training

        log = {'train/actor_loss': actor_loss,
               'train/entropy_loss': entropy_loss,
               'train/mirror_loss': mirror_loss,
               'train/critic_loss': critic_loss,
               'train/kl_divergence': kl,
               'misc/total_steps': total_steps,
               'misc/trajectory_count': trajectory_count,
               'misc/num_batches': num_batches,
               'misc/iterations': iterations,
               'misc/train_epoch': train_epoch,
               'misc/time_training': time_training.total_seconds(),
               'misc/time_elapsed': (datetime.datetime.now() - time_now).total_seconds()}

        checkpoint = {
            'total_steps': total_steps,
            'iterations': iterations,
            'trajectory_count': trajectory_count,
            'actor_state_dict': agent.actor.state_dict(),
            'critic_state_dict': agent.critic.state_dict(),
            'optimizer_actor_state_dict': agent.optimizer_actor.state_dict(),
            'optimizer_critic_state_dict': agent.optimizer_critic.state_dict(),

        }
        checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint-{run.name}.pt')
        torch.save(checkpoint, checkpoint_path)
        run.save(checkpoint_path, policy='now')

        if total_steps - prev_save_steps >= args.model_save_steps:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint-{run.name}-{iterations}.pt')
            torch.save(checkpoint, checkpoint_path)

            run.save(checkpoint_path, policy='now')

            prev_save_steps = total_steps

            log['misc/checkpoint_saved_iteration'] = iterations

        if mean_train_rewards >= max_reward:
            max_reward = mean_train_rewards
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint-best-{run.name}.pt')
            torch.save(agent.actor.state_dict(), checkpoint_path)
            run.save(checkpoint_path, policy='now')

        logging.debug(log)
        run.log(log, step=total_steps)

        iterations += 1

    pbar_total_steps.close()
    pbar_evaluator.close()
    [pc.close() for pc in pbar_collector]
    logging.info('Training done')
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO")

    # Training
    parser.add_argument("--model_save_steps", type=int, default=int(1e8), help="Save model steps")
    parser.add_argument("--max_steps", type=int, default=int(100e9), help="Maximum number of training steps")
    parser.add_argument("--num_epoch", type=int, default=10, help="PPO parameter")
    parser.add_argument("--evaluate_freq", type=int, default=2,
                        help="Policy evaluation frequency")
    parser.add_argument("--n_collectors", type=int, default=80, help="Number of collectors")
    parser.add_argument("--n_evaluators", type=int, default=4, help="Number of evaluators")
    parser.add_argument("--buffer_size", type=int, default=50000, help="Buffer size for training")
    parser.add_argument("--eval_buffer_size", type=int, default=3000, help="Buffer size for evaluation")
    parser.add_argument("--mini_batch_size", type=int, default=32, help="Minibatch size")
    parser.add_argument("--empty_cuda_cache", action='store_true', help="Whether to empty cuda cache")
    parser.add_argument("--time_horizon", type=int, default=500,
                        help="The maximum length of the episode")
    parser.add_argument('--num_cassie_prob', type=float, nargs='+', default=[1.0],
                        help='Probability of number of cassie')
    parser.add_argument("--reward_name", type=str, default='feet_air_time', help="Name of the reward function")
    parser.add_argument("--position_offset", type=float, default=0.2, help="Cassiepede position offset")
    parser.add_argument("--poi_heading_range", type=float, default=0.0, help="Poi heading range")
    parser.add_argument("--poi_position_offset", type=float, default=0.0, help="Poi offset from cassie")
    parser.add_argument("--perturbation_force", type=float, help="Force to apply to the deck", default=0)
    parser.add_argument("--perturbation_torque", type=float, help="Torque to apply to the deck", default=0)
    parser.add_argument("--force_prob", type=float, help="Prob of force to apply to the deck", default=0.0)
    parser.add_argument("--cmd_noise", type=float,
                        help="Noise to cmd for each cassie. Tuple of 3 (x_vel, y_vel, turn_rate (deg/t))", nargs=3,
                        default=[0.0, 0.0, 0.0])
    parser.add_argument("--cmd_noise_prob", type=float, help="Prob of noise added to cmd for each cassie", default=0.0)
    parser.add_argument("--mask_tarsus_input", action='store_true', help="Mask tarsus input with zeros")
    parser.add_argument("--device", type=str, default='cuda', help="Device name")
    parser.add_argument("--state_history_size", type=int, default=50,
                        help="The size of state history to return from env")
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')

    # Network
    parser.add_argument("--hidden_dim", type=int, required=False, help="Latent dim of non-proprioception state")
    parser.add_argument("--lstm_hidden_dim", type=int, required=False, help="Number of hidden units in LSTM")
    parser.add_argument('--lstm_num_layers', type=int, required=False, help='Number of layers in LSTM')
    parser.add_argument("--transformer_hidden_dim", type=int, required=False,
                        help="Number of hidden units in Transformer")
    parser.add_argument('--transformer_num_layers', type=int, required=False,
                        help='Number of layers in transformer encoder')
    parser.add_argument('--transformer_num_heads', type=int, required=False,
                        help='Number of attention heads in transformer encoder')
    parser.add_argument('--transformer_dim_feedforward', type=int, required=False,
                        help='Feedforward dimension in transformer')
    parser.add_argument('--actor_name', type=str, required=True, help='Name of actor class')
    parser.add_argument('--critic_name', type=str, required=True, help='Name of critic class')

    # Optimizer
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--set_adam_eps", action='store_true', help="Set Adam epsilon=1e-5")
    parser.add_argument("--eps", type=float, default=1e-5, help="Eps of Adam optimizer (default: 1e-5)")
    parser.add_argument("--std", type=float, help="Std for action")
    parser.add_argument("--use_orthogonal_init", action='store_true', help="Orthogonal initialization")
    parser.add_argument("--seed", type=int, default=0, help="Seed")

    # PPO
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=1.0, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--kl_check", action='store_true', help="Whether to check kl divergence")
    parser.add_argument("--kl_threshold", type=float, default=0.2, help="KL threshold of early stopping")
    parser.add_argument("--kl_check_min_itr", type=int, default=2,
                        help="Epoch after which kl check is done")
    parser.add_argument("--use_adv_norm", action='store_true', help="Advantage normalization")
    parser.add_argument("--use_reward_scaling", action='store_true', help="Reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Policy entropy")
    parser.add_argument("--use_lr_decay", action='store_true', help="Learning rate Decay")
    parser.add_argument("--use_grad_clip", action='store_true', help="Gradient clip")
    parser.add_argument("--grad_clip", type=str, default=0.05, help="Gradient clip value")
    # parser.add_argument("--use_mirror_loss", action='store_true', help="Whether to use mirror loss")
    parser.add_argument("--mirror_loss", choices=['supervised', 'ppo'], nargs='*', default=[])

    # Wandb
    parser.add_argument("--project_name", type=str, default='roadrunner_cassiepede', help="Name of project")
    parser.add_argument("--previous_run", type=str, default=None, help="Name of previous run")
    parser.add_argument("--parent_run", type=str, default=None, help="Name of parent run")
    parser.add_argument("--previous_checkpoint", type=str, default=None, help="Timestep of bootstrap checkpoint")
    parser.add_argument("--wandb_mode", type=str, default='online', help="Wandb mode")

    args = parser.parse_args()

    ray.init(num_cpus=args.n_collectors + args.n_evaluators, local_mode=False)

    main()

# Example Run script
# export PYTHONPATH=.
# export WANDB_API_KEY=
# python algo/cassiepede/training.py \
#   --n_collectors 120 \
#   --n_evaluators 6 \
#   --time_horizon 500 \
#   --buffer_size 60000 \
#   --eval_buffer_size 3000 \
#   --evaluate_freq 4 \
#   --num_epoch 5 \
#   --mini_batch_size 32 \
#   --hidden_dim 64 \
#   --lstm_hidden_dim 64 \
#   --lstm_num_layers 2 \
#   --use_orthogonal_init \
#   --set_adam_eps \
#   --kl_check \
#   --kl_check_min_itr 2 \
#   --use_adv_norm \
#   --use_lr_decay \
#   --use_grad_clip \
#   --reward_name locomotion_cassiepede_feetairtime_modified \
#   --project_name roadrunner_cassiepede \
#   --wandb_mode online \
#   --device cuda:0 \
#   --position_offset 1.0 \
#   --poi_heading_range 1.05 \
#   --gamma 0.95 \
#   --std 0.13 \
#   --entropy_coef 0.01 \
#   --num_cassie_prob 0.2 0.8 \
#   --wandb_mode online \
#   --perturbation_force 30.0 \
#   --force_prob 0.2 \
#   --cmd_noise 0.0 0.0 0.0 \
#   --cmd_noise_prob 0.0
