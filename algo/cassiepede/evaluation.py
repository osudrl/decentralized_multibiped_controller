import copy
import copy
import time

import numpy as np
import torch


from env.util.quaternion import quaternion2euler
from util.mirror import mirror_tensor


from env.cassie.cassiepede.cassiepede import Cassiepede

import sys
import tty
import select
from algo.common.utils import *
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)


def apply_linear_perturbation(env, magnitude):
    if magnitude == 0:
        # Save computation
        env.force_vector[:] = 0.0, 0.0, 0.0
    else:
        theta = np.random.uniform(-np.pi, np.pi)  # polar angle

        z_force_limit = 5.0

        # Compute azimuthal delta for limited force in upward direction
        azimuth_delta = np.arccos(np.minimum(1, z_force_limit / magnitude))

        phi = np.random.uniform(azimuth_delta, np.pi)  # azimuthal angle

        x = magnitude * np.sin(phi) * np.cos(theta)
        y = magnitude * np.sin(phi) * np.sin(theta)
        z = magnitude * np.cos(phi)

        # Override env's force vector just for visualization purpose
        env.force_vector[:] = x, y, z

    for i in range(len(env.target_pertub_bodies)):
        env.sim.data.xfrc_applied[env.target_pertub_bodies[i], :3] = env.force_vector[i]


def main():
    if torch.backends.mps.is_available():
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    keyboard = True
    offscreen = False
    legacy_actor = False
    dummy_actor = False
    vis_center_of_mass = False
    vis_attention_weight = False
    vis_depth_image = False
    done_enable = False
    compute_mirror_loss = False
    single_windowed_force = None
    centralized_policy = False
    record_video = False

    custom_terrain = None

    sampling_freq = np.array(args.num_cassie_prob, dtype=float) / np.sum(args.num_cassie_prob)

    num_cassie = np.random.choice(np.arange(1, len(sampling_freq) + 1), p=sampling_freq)
    env = Cassiepede(
        reward_name=args.reward_name,
        simulator_type='mujoco',
        policy_rate=50,
        dynamics_randomization=True,
        # state_noise=[0.05, 0.1, 0.01, 0.2, 0.01, 0.2, 0.05, 0.05, 0.05],
        state_noise=0,
        velocity_noise=0.0,
        state_est=False,
        integral_action=False,
        com_vis=vis_center_of_mass,
        depth_input=vis_depth_image,
        num_cassie=num_cassie,
        custom_terrain=custom_terrain,
        poi_position_offset=0.0 if num_cassie == 1 else args.poi_position_offset,
        perturbation_force=(not single_windowed_force) * args.perturbation_force,
        force_prob=(not single_windowed_force) * args.force_prob,
        only_deck_force=False,
        height_control=True,
        position_offset=0.0 if num_cassie == 1 else args.position_offset,
        poi_heading_range=0.0 if num_cassie == 1 else args.poi_heading_range,
        cmd_noise=args.cmd_noise,
        merge_states=centralized_policy,
        state_history_size=args.state_history_size,
        cmd_noise_prob=args.cmd_noise_prob,
        mask_tarsus_input=args.mask_tarsus_input,
        enable_hfield=args.terrain >= 0,
        hfield_idx=args.terrain,
        offscreen=offscreen)

    env.eval(True)

    args.state_dim = env.observation_size
    args.action_dim = env.action_size

    if keyboard:
        tty.setcbreak(sys.stdin.fileno())

    actors = []
    for run_name in args.runs_name:
        args_ = copy.deepcopy(args)
        args_.run_name = run_name

        actor = load_actor(args_, device)

        print(actor)

        print('actor', sum(p.numel() for p in actor.parameters()))

        actors.append(actor)

    batch_size = (1 if len(actors) > 1 else (1 if env._merge_states else env.num_cassie))

    done = np.zeros(env.num_cassie, dtype=bool)
    episode_length = 0
    episode_reward = defaultdict(lambda: 0)

    # This is penalty (before kernel applied to reward)
    episode_reward_raw = defaultdict(lambda: 0)

    for actor in actors:
        if hasattr(actor, 'init_hidden_state'):
            actor.init_hidden_state(device=device, batch_size=batch_size * (1 + compute_mirror_loss))

    state = env.reset(interactive_evaluation=args.evaluation_mode != 'random')

    mirror_dict = env.get_mirror_dict()

    for k in mirror_dict['state_mirror_indices'].keys():
        mirror_dict['state_mirror_indices'][k] = torch.tensor(mirror_dict['state_mirror_indices'][k],
                                                              dtype=torch.float32,
                                                              device=device)

    mirror_dict['action_mirror_indices'] = torch.tensor(mirror_dict['action_mirror_indices'],
                                                        dtype=torch.float32,
                                                        device=device)

    match args.evaluation_mode:
        case 'interactive':
            env.x_velocity_poi = np.zeros(env.num_cassie, dtype=float)
            env.y_velocity_poi = np.zeros(env.num_cassie, dtype=float)
            env.turn_rate_poi = np.zeros(env.num_cassie, dtype=float)
            env.height_base = np.full(env.num_cassie, 0.75, dtype=float)
        case 'random':
            pass
        case _:
            raise NotImplementedError

    render_state = offscreen

    if not offscreen:
        env.sim.viewer_init(record_video=record_video, overlay_text=("Wandb run(s) name:", ", ".join(args.runs_name)))
        render_state = env.sim.viewer_render()

    done_sum = np.zeros(env.num_cassie)

    poi_idx = 0

    reset = False

    total_reward = 0

    total_power = np.zeros(env.num_cassie, dtype=float)

    initial_poi_position = env.get_poi_position().copy()

    if any(['Transformer' in actor.__class__.__name__ for actor in actors]):
        if len(actors) > 1:
            src_key_padding_mask = torch.zeros(1, 1, 1, dtype=torch.bool)
        else:
            src_key_padding_mask = torch.zeros(1, 1, env.num_cassie, dtype=torch.bool)

    while render_state:
        start_time = time.time()
        if offscreen or not env.sim.viewer_paused():

            state_ = OrderedDict()
            # Numpy to tensor
            for k in state.keys():
                state_[k] = torch.tensor(state[k], dtype=torch.float32, device=device)

                state_[k] = state_[k].unsqueeze(0).unsqueeze(0)

            if compute_mirror_loss:
                for k in state_.keys():
                    s_mirrored = mirror_tensor(state_[k], mirror_dict['state_mirror_indices'][k])

                    s_mirrored_mirrored = mirror_tensor(s_mirrored, mirror_dict['state_mirror_indices'][k])

                    assert torch.allclose(state_[k], s_mirrored_mirrored), f"Mirror tensor failed for {k}"

                    state_[k] = torch.cat([state_[k], s_mirrored], dim=0)

            actions = []

            for i, actor in enumerate(actors):
                if actor is None:
                    action = np.random.uniform(-0.2, 0.2, args.action_dim)
                else:
                    with torch.inference_mode():
                        if not compute_mirror_loss:
                            if len(actors) > 1:
                                state__ = OrderedDict((k, v[:, :, [i]]) for k, v in state_.items())
                                if 'Transformer' in actor.__class__.__name__:
                                    action, *_ = actor.forward(state__, src_key_padding_mask=src_key_padding_mask,
                                                               need_weights=vis_attention_weight)
                                else:
                                    action, _ = actor.forward(state__)
                            else:
                                if 'Transformer' in actor.__class__.__name__:
                                    action, *_ = actor.forward(state_, src_key_padding_mask=src_key_padding_mask,
                                                               need_weights=vis_attention_weight)
                                else:
                                    action, _ = actor.forward(state_)

                            action = action.cpu().numpy().squeeze(0).squeeze(0)
                        else:
                            if len(actors) > 1:
                                state__ = OrderedDict((k, v[:, :, [i, i + env.num_cassie]])
                                                      for k, v in state_.items())

                                if 'Transformer' in actor.__class__.__name__:
                                    action, _ = actor.forward(state__, src_key_padding_mask=src_key_padding_mask)
                                else:
                                    action, _ = actor.forward(state__)

                                mirrored_action = mirror_tensor(action[-1:], mirror_dict['action_mirror_indices'])
                                action = action[:1]
                            else:
                                if 'Transformer' in actor.__class__.__name__:
                                    action, _ = actor.forward(state_, src_key_padding_mask=src_key_padding_mask)
                                else:
                                    action, _ = actor.forward(state_)

                                mirrored_action = mirror_tensor(action[-env.num_cassie:],
                                                                mirror_dict['action_mirror_indices'])

                                action = action[:env.num_cassie]

                            mirror_loss = (0.5 * F.mse_loss(action, mirrored_action)).item()

                            logging.info(f'Mirror loss: {mirror_loss}')

                            print('Mirror action shape:', mirrored_action.shape)

                            if 'Transformer' in actor.__class__.__name__:
                                action = mirrored_action.cpu().numpy().squeeze(0).squeeze(0)
                            else:
                                action = mirrored_action.cpu().numpy().squeeze(0).squeeze(0)

                actions.append(action)

            action = np.concatenate(actions, axis=0)

            if single_windowed_force:
                if episode_length == single_windowed_force[0]:
                    apply_linear_perturbation(env, args.perturbation_force)
                elif episode_length == single_windowed_force[1]:
                    apply_linear_perturbation(env, 0)

            state, reward, done, info = env.step(action)

            total_reward += reward

            for i in range(env.num_cassie):
                if 'reward_raw' in info:
                    for reward_key, reward_val in info['reward_raw'][i].items():
                        episode_reward_raw[reward_key] += reward_val

                if 'reward' in info:
                    for reward_key, reward_val in info['reward'][i].items():
                        episode_reward[reward_key] += reward_val

            done_sum += done

            if keyboard and sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                input_char = sys.stdin.read(1)
                print('input_char:', input_char)
                match input_char:
                    case "a":
                        env.y_velocity_poi += 0.1
                        print('y_velocity_poi:', env.y_velocity_poi)
                    case "d":
                        env.y_velocity_poi -= 0.1
                        print('y_velocity_poi:', env.y_velocity_poi)
                    case "w":
                        env.x_velocity_poi += 0.1
                        print('x_velocity_poi:', env.x_velocity_poi)
                    case "s":
                        env.x_velocity_poi -= 0.1
                        print('x_velocity_poi:', env.x_velocity_poi)
                    case "q":
                        env.turn_rate_poi += np.radians(5)
                        print('turn_rate_poi:', env.turn_rate_poi)
                    case "e":
                        env.turn_rate_poi -= np.radians(5)
                        print('turn_rate_poi:', env.turn_rate_poi)
                    case "1":
                        env.height_base -= 0.05
                        print('height_base:', env.height_base)
                    case "2":
                        env.height_base += 0.05
                        print('height_base:', env.height_base)
                    case 'h':
                        # soft reset
                        for actor in actors:
                            if hasattr(actor, 'init_hidden_state'):
                                actor.init_hidden_state(device=device,
                                                        batch_size=batch_size * (1 + compute_mirror_loss))
                        print('Hidden state reset')

                    case input_char if input_char in ['i', 'm', 'j', 'k', 'l', 'u', 'o']:

                        if input_char in ['u', 'o']:
                            delta_rot = {'u': -0.1, 'o': 0.1}[input_char]
                            delta_pos = [0, 0]
                        else:
                            delta_pos = {'i': [0.1, 0],
                                         'm': [-0.1, 0],
                                         'j': [0, 0.1],
                                         'k': [0, -0.1],
                                         'l': [0, 0]}[input_char]
                            delta_rot = 0

                        # Get local cassie positions. We cannot use sensor as it gives global position
                        local_base_position = []
                        for i in range(env.num_cassie):
                            i = '' if i == 0 else f'c{i + 1}_'
                            local_base_position.append(env.sim.model.body(f'{i}cassie-pelvis').pos[:2] + delta_pos)
                        local_base_position = np.array(local_base_position)

                        # Get local poi position and orientation. We cannot use sensor as it gives global position
                        if input_char == 'l':
                            poi_position = local_base_position[poi_idx]
                            poi_idx = (poi_idx + 1) % env.num_cassie
                        else:
                            poi_position = env.sim.model.site('poi_site').pos[:2] + delta_pos
                        poi_orientation = \
                            quaternion2euler(np.array(env.sim.model.site('poi_site').quat).reshape(1, -1))[0, -1] \
                            + delta_rot

                        # Update the poi position
                        env._update_poi_position(poi_position=poi_position, poi_orientation=poi_orientation)

                        # Re compute encoding
                        env._compute_encoding(poi_position=poi_position, poi_orientation=poi_orientation,
                                              base_positions=local_base_position)

                        # plotter_data['clear'] = True

                    case "r":
                        reset = True
      
            episode_length += 1

        if not offscreen:
            render_state = env.sim.viewer_render()

            delaytime = max(0, env.default_policy_rate / 2000 - (time.time() - start_time))

            time.sleep(delaytime)

        if done.any():
            print('done', done, info.get('done_info', None))

        if (done.any() and done_enable) or reset or episode_length >= args.time_horizon:
            reset = False
            poi_idx = 0

            logging.info(
                f'Total reward dict:{episode_reward}\n'
                f'Total reward raw:{episode_reward_raw}\n'
                f'Total reward (all cassie): {np.sum(list(episode_reward.values()))}\n'
                f'Total reward:{total_reward}\nEpisode length:{episode_length}\n'
                f'Encoding: {env.encoding}\n'
                f'force_vector={(env.force_vector, np.linalg.norm(env.force_vector)) if hasattr(env, "force_vector") else None}\n'
                f'torque_vector={(env.torque_vector, np.linalg.norm(env.torque_vector)) if hasattr(env, "force_vector") else None}\n'
                f'Total power per step={total_power / episode_length}\n'
                f'done_info={info.get("done_info", None)}\n'
                f'single_windowed_force={single_windowed_force}\n')

            logging.info(f'odometry: {env.get_poi_position() - initial_poi_position}')

            rem_rotation = (args.time_horizon - episode_length) * env.turn_rate_poi[0] / env.default_policy_rate

            logging.info(
                f'orientation error: {np.degrees((quaternion2euler(np.array([env.get_poi_orientation()]))[0, -1] - env.orient_add[0] - rem_rotation + np.pi) % (2 * np.pi) - np.pi)}, rem_rotation: {np.degrees(rem_rotation)} env.orient_add[0]={np.degrees(env.orient_add[0])}, current_orientation={np.degrees(quaternion2euler(np.array([env.get_poi_orientation()]))[0, -1])}')

            episode_reward.clear()
            episode_reward_raw.clear()

            time.sleep(0.5)

            total_reward = 0

            total_power = np.zeros(env.num_cassie, dtype=float)

            done = np.zeros(env.num_cassie, dtype=bool)
            done_sum = np.zeros(env.num_cassie)

            state = env.reset(interactive_evaluation=args.evaluation_mode != 'random')

            initial_poi_position = env.get_poi_position().copy()

            if args.evaluation_mode == 'interactive':
                env.x_velocity_poi = np.zeros(env.num_cassie, dtype=float)
                env.y_velocity_poi = np.zeros(env.num_cassie, dtype=float)
                env.turn_rate_poi = np.zeros(env.num_cassie, dtype=float)
                env.height_base = np.full(env.num_cassie, 0.75, dtype=float)

            print('commands:', env.x_velocity_poi, env.y_velocity_poi, env.turn_rate_poi)
            episode_length = 0
            episode_reward.clear()

            for actor in actors:
                if hasattr(actor, 'init_hidden_state'):
                    actor.init_hidden_state(device=device,
                                            batch_size=batch_size * (1 + compute_mirror_loss))

            if not offscreen and hasattr(env.sim, 'renderer'):
                if env.sim.renderer is not None:
                    print("re-init non-primary screen renderer")
                    env.sim.renderer.close()
                    env.sim.init_renderer(offscreen=env.offscreen,
                                          width=env.depth_image_dim[0], height=env.depth_image_dim[1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO")

    parser.add_argument('--time_horizon', type=int, default=1e5)

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
    
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--project_name', type=str, default='roadrunner_cassiepede')
    parser.add_argument('--runs_name', type=str, nargs='+', default=["2024-02-16 02:41:50.685098"])
    parser.add_argument('--num_cassie_prob', nargs='+', type=float, default=[0.0, 1.0, 0.0])
    parser.add_argument('--model_checkpoint', type=str, default="latest")
    parser.add_argument('--reward_name', type=str, default='locomotion_cassiepede')
    parser.add_argument("--position_offset", type=float, default=0.2, help="Cassiepede position offset")
    parser.add_argument("--poi_heading_range", type=float, default=0.0, help="Poi heading range")
    parser.add_argument("--poi_position_offset", type=float, default=0.0, help="Poi offset from cassie")
    parser.add_argument("--perturbation_force", type=float, help="Force to apply to the deck", default=0)
    parser.add_argument("--perturbation_torque", type=float, help="Torque to apply to the deck", default=0)
    parser.add_argument("--force_prob", type=float, help="Prob of force to apply to the deck", default=0.0)
    parser.add_argument("--cmd_noise", type=float,
                        help="Noise to cmd for each cassie. Tuple of 3 (x_vel, y_vel, turn_rate (deg/t))", nargs=3,
                        default=[0.0, 0.0, 0.0])
    parser.add_argument("--state_history_size", type=int, default=50,
                        help="The size of state history to return from env")

    parser.add_argument("--mask_tarsus_input", action='store_true', help="Mask tarsus input with zeros")
    parser.add_argument("--cmd_noise_prob", type=float, help="Prob of noise added to cmd for each cassie", default=0.0)
    parser.add_argument('--offline', action='store_true', help='Whether to load model from wandb. This requires WANDB_API_KEY to be set')
    parser.add_argument('--evaluation_mode', type=str, default='interactive',
                        choices=['interactive', 'random'])
    parser.add_argument('--terrain', type=int, default=-1)

    args = parser.parse_args()

    # Either supply one run name of all cassie or supply run name for each cassie
    assert len(args.runs_name) == 1 or len(args.num_cassie_prob) == len(args.runs_name)

    main()