from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env import VecEnvWrapper
import numpy as np
import pickle
import gym
import argparse
import os
from scipy import stats
import visdom

import torch
import utils
import TD3
import OurDDPG
import DDPG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_env(env_id, seed, rank):
    def _thunk():
        env = gym.make(env_id)

        env.seed(seed + rank)
        return env

    return _thunk


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = reward
        done = 1 * done
        return obs, reward, done, info


def make_vec_envs(env_name, num_processes):
    seed = np.random.randint(0, 10000000)
    envs = [
        make_env(env_name, seed, i)
        for i in range(num_processes)
    ]

    envs = SubprocVecEnv(envs)
    envs = VecPyTorch(envs, device)

    return envs


def moving_average(a, n=3):
    plot_data = np.zeros_like(a)
    for idx in range(len(a)):
        length = min(idx, n)
        plot_data[idx] = a[idx - length:idx + 1].mean()
    return plot_data


def vis_plot(viz, log_dict):
    ma_length = 0
    if viz is not None:
        for field in log_dict:
            if len(log_dict[field]) > 0:
                _, values = zip(*log_dict[field])

                plot_data = np.array(log_dict[field])
                viz.line(X=plot_data[:, 0], Y=moving_average(plot_data[:, 1], ma_length), win=field,
                         opts=dict(title=field, legend=[field]))


def swap_policies_if_better(policy, r, std, target_r, target_std, force_swap, criterion='prob', significance=0.05,
                            n=10, cpg=False):
    if target_std is not None and std is not None:
        if target_std > 0 or std > 0:
            DF_1 = (std ** 2 / n + target_std ** 2 / n) ** 2
            DF_2 = ((std ** 2 / n) ** 2) / (n - 1)
            DF_3 = ((target_std ** 2 / n) ** 2) / (n - 1)
            DF = int(DF_1 / (DF_2 + DF_3))
            t = (r - target_r) / np.sqrt((std ** 2 / n) + (target_std ** 2 / n))
            p = stats.t.cdf(t, df=DF)
        else:
            t = 0
            if r > target_r:
                p = 1
            else:
                p = 0
    else:
        t = 0
        if target_r is None or r > target_r:
            p = 1
        else:
            p = 0

    swapped = False
    if criterion == 'prob':
        if significance > 1 - p or force_swap:
            swapped = True
    elif criterion == 'max':
        if target_r is None or r > target_r or force_swap:
            swapped = True
    elif criterion == 'margin':
        if r > target_r * 1.05 or force_swap:
            swapped = True
    elif criterion == 'always':
        swapped = True
    else:
        raise NotImplementedError

    if swapped:
        if cpg:
            policy.actor_target.load_state_dict(policy.actor.state_dict())
        policy.actor_static.load_state_dict(policy.actor.state_dict())
    if target_r is None:
        target_r = 0
        target_std = 0
    print("R: %f, STD: %f, Target R: %f, Target STD: %f, t-statistic: %f, p: %f, swapped: %s" % (
        r, std, target_r, target_std, t, p, str(swapped)))
    return 1 if swapped else 0


# Runs policy for X episodes and returns average reward
def evaluate_policy(policy, memory, eval_episodes=1, force_swap=False, cpg=False, criterion='eval_prob',
                    significance=0.05, prev_r=None, prev_std=None, reevaluate=True, swap=True):
    additional_samples = 0

    if len(memory.storage) == 0:
        train = False
    else:
        train = True

    num_evals = max(eval_envs, eval_episodes)
    if criterion is not None and ('eval' in criterion or 'replay' in criterion):
        use_memory = 'replay' in criterion
        criterion = criterion.split('_')[1]
    else:
        use_memory = False

    with torch.no_grad():
        rewards = np.zeros(num_evals)
        target_rewards = np.zeros(num_evals)
        total_steps = np.zeros(num_evals)

        for iter in range(int(num_evals / eval_envs)):
            not_done_vec = np.ones(eval_envs)
            valid_idx = np.ones(eval_envs)
            obs = eval_env.reset()
            while np.sum(not_done_vec) > 0:
                action = policy.actor(obs) * max_action
                new_obs, reward, done, _ = eval_env.step(action)

                if swap and criterion is not None:
                    # Store data in replay buffer
                    for idx in range(eval_episodes):
                        if valid_idx[idx] == 1:
                            additional_samples += 1
                            done_bool = 0 if total_steps[idx] + 1 == env._max_episode_steps else float(done[idx])
                            memory.add((obs[idx].cpu().data.numpy().flatten(), new_obs[idx].cpu().data.numpy().flatten(), action[idx].cpu().data.numpy().flatten(), reward[idx].flatten()[0], done_bool, 1))

                            if done[idx]:
                                valid_idx[idx] = 0
                obs = new_obs

                rewards[iter * eval_envs:(iter + 1) * eval_envs] += reward * not_done_vec
                total_steps[iter * eval_envs:(iter + 1) * eval_envs] += not_done_vec

                not_done_vec = not_done_vec * (1 - done)
            avg_reward = np.mean(rewards[:eval_episodes])
            std = np.std(rewards[:eval_episodes])

        if not use_memory:
            total_target_steps = np.zeros(num_evals)
            for iter in range(int(num_evals / eval_envs)):
                not_done_vec = np.ones(eval_envs)
                valid_idx = np.ones(eval_envs)
                obs = eval_env.reset()
                while np.sum(not_done_vec) > 0:
                    additional_samples += 1
                    action = policy.actor_static(obs) * max_action

                    new_obs, reward, done, _ = eval_env.step(action)

                    if swap and reevaluate and criterion is not None:
                        # Store data in replay buffer
                        for idx in range(eval_episodes):
                            if valid_idx[idx] == 1:
                                additional_samples += 1
                                done_bool = 0 if total_target_steps[idx] + 1 == env._max_episode_steps else float(done[idx])
                                memory.add((obs[idx].cpu().data.numpy().flatten(), new_obs[idx].cpu().data.numpy().flatten(), action[idx].cpu().data.numpy().flatten(), reward[idx].flatten()[0], done_bool, 1))

                                if done[idx]:
                                    valid_idx[idx] = 0
                    obs = new_obs

                    target_rewards[iter * eval_envs:(iter + 1) * eval_envs] += reward * not_done_vec
                    total_target_steps[iter * eval_envs:(iter + 1) * eval_envs] += not_done_vec

                    not_done_vec = not_done_vec * (1 - done)
            if reevaluate:
                avg_target_reward = np.mean(target_rewards[:eval_episodes])
                target_std = np.std(target_rewards[:eval_episodes])
            else:
                avg_target_reward = prev_r
                target_std = prev_std
        else:
            distribution = stats.norm(0, 0.1)
            ind = np.ones(len(memory.storage))
            ind = np.cumsum(ind) - 1
            x, y, u, r, d, p = [], [], [], [], [], []

            for i in ind:
                X, Y, U, R, D, P = memory.storage[int(i)]
                x.append(np.array(X, copy=False))
                y.append(np.array(Y, copy=False))
                u.append(np.array(U, copy=False))
                r.append(np.array(R, copy=False))
                d.append(np.array(D, copy=False))
                p.append(np.array(P, copy=False))

            x, y, u, r, d, p = np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(
                -1, 1), np.array(p).reshape(len(memory.storage), -1)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)

            distances = (action - policy.actor(state) * max_action).cpu().numpy()
            target_distances = (action - policy.actor_static(state) * max_action).cpu().numpy()
            probs = distribution.pdf(-np.abs(distances))
            target_probs = distribution.pdf(-np.abs(target_distances))

            is_rewards = []
            is_target_rewards = []
            idx = 0

            is_reward = 0
            is_target_reward = 0
            is_prev_prob = 1
            is_prob = 1
            is_target_prob = 1
            while idx < len(memory.storage):
                is_prev_prob *= np.prod(p[idx])
                is_prob *= np.prod(probs[idx])
                is_target_prob *= np.prod(target_probs[idx])

                if is_prev_prob == 0:
                    is_reward = 0
                    is_target_reward = 0
                    is_prob = 0
                    is_target_prob = 0
                else:
                    is_reward += (is_prob / is_prev_prob) * r[idx]
                    is_target_reward += (is_target_prob / is_prev_prob) * r[idx]

                if d[idx] == 1:
                    print('Probs: ' + str(is_prev_prob) + ' ' + str(is_prob) + ' ' + str(is_target_prob))
                    if is_prev_prob > 0.5:
                        if is_prob > 0.5:
                            is_rewards.append(is_reward)
                        if is_target_prob > 0.5:
                            is_target_rewards.append(is_target_reward)

                    is_reward = 0
                    is_target_reward = 0
                    is_prev_prob = 1
                    is_prob = 1
                    is_target_prob = 1
                idx += 1

            avg_reward = np.mean(is_rewards)
            std = np.std(is_rewards)
            avg_target_reward = np.mean(is_target_rewards)
            target_std = np.std(is_target_rewards)

    print("---------------------------------------")
    print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    if criterion is not None:
        swapped = swap_policies_if_better(policy, avg_reward, std, avg_target_reward, target_std, force_swap, criterion,
                                          significance, eval_episodes, cpg)
        if swapped or prev_r is None:
            prev_r = avg_reward
            prev_std = std
        else:
            prev_r = avg_target_reward
            prev_std = target_std
    else:
        swapped = False
    print("---------------------------------------")
    if train:
        policy.train(memory, additional_samples,
                     args.batch_size,
                     args.discount, args.tau, args.policy_noise,
                     args.noise_clip,
                     args.policy_freq,
                     update_target_actor=criterion is None or not cpg,
                     update_target_q=not args.swap_q)
    return np.mean(rewards[eval_episodes:]), np.mean(target_rewards[eval_episodes:]), swapped, additional_samples, prev_r, prev_std


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name", default="TD3-swap", choices=['TD3-swap', 'TD3'])  # Policy name
    parser.add_argument("--env_name", default="HalfCheetah-v2")  # OpenAI gym environment name
    parser.add_argument("--start_timesteps", default=1e4,
                        type=int)  # How many time steps purely random policy is run for
    parser.add_argument("--swap_start", default=0, type=int)
    parser.add_argument('--swap_q', default=False, action='store_true')
    parser.add_argument("--swap_freq", default=1e3, type=float)  # How often (time steps) we evaluate and swap
    parser.add_argument('--swap_criterion', default='replay_prob',
                        choices=['replay_max', 'replay_prob', 'eval_max', 'eval_prob', 'always'])
    parser.add_argument('--no_reevaluation', default=False, action='store_true')
    parser.add_argument("--eval_freq", default=1e3, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=float)  # Max time steps to run environment for
    parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=100, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument('--visualize', default=False, action='store_true')
    parser.add_argument('--no_target_q', default=False, action='store_true')
    parser.add_argument('--cpg', default=False, action='store_true')
    parser.add_argument('--experiment_name', default=None, type=str,
                        help='For multiple different experiments, provide an informative experiment name')
    parser.add_argument("--eval_episodes", default=10, type=int)  # Number of episodes for policy evaluation
    parser.add_argument("--target_distance_weight", default=0, type=float)  # Frequency of delayed policy updates
    parser.add_argument("--significance", default=0.1, type=float)  # Significance for the probability based approach
    args = parser.parse_args()
    args.target_q = not args.no_target_q
    args.reevaluate = not args.no_reevaluation

    base_dir = os.getcwd() + '/models/' + args.policy_name
    if args.swap_q:
        base_dir = base_dir + '_swap_q'
    if args.experiment_name is not None:
        base_dir += '/' + args.experiment_name
    base_dir = base_dir + '/' + args.swap_criterion + '/' + args.env_name + '/'

    run_number = 0
    while os.path.exists(base_dir + str(run_number)):
        run_number += 1
    base_dir = base_dir + str(run_number)
    os.makedirs(base_dir)

    results_dict = {'eval_rewards': [],
                    'target_eval_rewards': [],
                    'value_losses': [],
                    'policy_losses': [],
                    'train_rewards': [],
                    'swapped': [],
                    'actor_loss': [],
                    'critic_loss': [],
                    'additional_samples': []
                    }

    if args.visualize:
        vis = visdom.Visdom(env=base_dir)
    else:
        vis = None

    env = gym.make(args.env_name)

    eval_envs = 100 + args.eval_episodes
    eval_env = make_vec_envs(args.env_name, eval_envs)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    if args.policy_name == 'TD3':
        args.swap_criterion = None

    # Initialize policy
    if args.policy_name == "TD3" or args.policy_name == 'TD3-swap':
        policy = TD3.TD3(state_dim, action_dim, 1, args.target_q, args.target_distance_weight)
    elif args.policy_name == "OurDDPG":
        policy = OurDDPG.DDPG(state_dim, action_dim, 1)
    elif args.policy_name == "DDPG":
        policy = DDPG.DDPG(state_dim, action_dim, 1)
    else:
        raise NotImplementedError

    replay_buffer = utils.ReplayBuffer()

    total_timesteps = 0
    total_timesteps_with_eval = 0
    timesteps_since_eval = 0
    timesteps_since_swapped = 0
    episode_num = 0
    done = True
    episode_reward = 0
    episode_timesteps = 0
    prev_r = None
    prev_std = None

    # Evaluate untrained policy
    eval_r, target_eval_r, swapped, additional_samples, prev_r, prev_std = evaluate_policy(policy,
                                                                                           memory=replay_buffer,
                                                                                           force_swap=False,
                                                                                           criterion=None,
                                                                                           cpg=args.cpg,
                                                                                           eval_episodes=args.eval_episodes,
                                                                                           significance=args.significance,
                                                                                           prev_r=prev_r,
                                                                                           prev_std=prev_std,
                                                                                           reevaluate=args.reevaluate,
                                                                                           swap='swap' in args.policy_name)
    results_dict['eval_rewards'].append((total_timesteps, eval_r))
    results_dict['target_eval_rewards'].append((total_timesteps, target_eval_r))
    results_dict['swapped'].append((total_timesteps, swapped))
    results_dict['additional_samples'].append((total_timesteps, 0))

    play_with_target = False
    distribution = stats.norm(0, 0.1)
    last_swapped = 0
    max_swap_diff = 20000

    while total_timesteps_with_eval < args.max_timesteps:
        if total_timesteps < args.swap_start or not args.cpg:
            swap_criterion = None
        else:
            swap_criterion = args.swap_criterion

        if done:
            play_with_target = not play_with_target if args.policy_name == 'TD3-swap' else False

            if total_timesteps != 0:
                print("%s - Run %d - Total T: %d Episode Num: %d Episode T: %d Reward: %f" % (
                    args.env_name, run_number, total_timesteps, episode_num, episode_timesteps, episode_reward))
                results_dict['train_rewards'].append((total_timesteps, episode_reward))
                if args.policy_name == "TD3" or args.policy_name == 'TD3-swap':
                    critic_loss, actor_loss = policy.train(replay_buffer, episode_timesteps, args.batch_size,
                                                           args.discount, args.tau, args.policy_noise, args.noise_clip,
                                                           args.policy_freq,
                                                           update_target_actor=swap_criterion is None or not args.cpg,
                                                           update_target_q=not args.swap_q)
                else:
                    policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau)
                    critic_loss = actor_loss = 0

                results_dict['actor_loss'].append((total_timesteps, actor_loss))
                results_dict['critic_loss'].append((total_timesteps, critic_loss))

            # Evaluate episode
            if timesteps_since_eval >= args.eval_freq:
                timesteps_since_eval %= args.eval_freq
                if timesteps_since_swapped >= args.swap_freq and total_timesteps > args.swap_start:
                    criterion = args.swap_criterion
                    timesteps_since_swapped %= args.swap_freq
                else:
                    criterion = None
                eval_r, target_eval_r, swapped, additional_samples, prev_r, prev_std = evaluate_policy(policy,
                                                                                                       memory=replay_buffer,
                                                                                                       force_swap=False,
                                                                                                       cpg=args.cpg,
                                                                                                       criterion=criterion,
                                                                                                       eval_episodes=args.eval_episodes,
                                                                                                       significance=args.significance,
                                                                                                       prev_r=prev_r,
                                                                                                       prev_std=prev_std,
                                                                                                       reevaluate=args.reevaluate,
                                                                                                       swap='swap' in args.policy_name)
                total_timesteps_with_eval += additional_samples
                if args.swap_q:
                    policy.critic_target.load_state_dict(policy.critic.state_dict())
                results_dict['eval_rewards'].append((total_timesteps, eval_r))
                results_dict['target_eval_rewards'].append((total_timesteps, target_eval_r))
                results_dict['swapped'].append((total_timesteps, swapped))
                results_dict['additional_samples'].append((total_timesteps, additional_samples))

                if swapped or swap_criterion is None:
                    last_swapped = total_timesteps
                elif last_swapped + max_swap_diff <= total_timesteps and args.cpg:
                    policy.actor.load_state_dict(policy.actor_static.state_dict())
                    params = policy.actor.state_dict()
                    for name in params:
                        if 'ln' in name:
                            pass
                        param = params[name]
                        param += torch.randn(param.shape).to(device) * 0.01

                    last_swapped = total_timesteps

                with open(base_dir + '/results', 'wb') as f:
                    pickle.dump(results_dict, f)

            # Reset environment
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

            vis_plot(vis, results_dict)

        # Select action randomly or according to policy
        if total_timesteps < args.start_timesteps:
            action = env.action_space.sample()
            prob = [0 for _ in range(action_dim)]
        else:
            if play_with_target:
                action = policy.select_action_target(np.array(obs))
            else:
                action = policy.select_action(np.array(obs))
            if args.expl_noise != 0:
                # Gaussian noise
                noise = np.random.normal(0, args.expl_noise, size=env.action_space.shape[0])
                action = (action + noise).clip(-1, 1) * max_action
                prob = distribution.pdf(-np.abs(noise))
            else:
                prob = 0

        # Perform action
        new_obs, reward, done, _ = env.step(action)
        done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)
        episode_reward += reward

        # Store data in replay buffer
        replay_buffer.add((obs, new_obs, action, reward, done_bool, prob))

        obs = new_obs

        episode_timesteps += 1
        total_timesteps += 1
        total_timesteps_with_eval += 1
        timesteps_since_eval += 1
        timesteps_since_swapped += 1

    # Final evaluation
    if timesteps_since_swapped >= args.swap_freq and total_timesteps > args.swap_start:
        criterion = args.swap_criterion
    else:
        criterion = None
    eval_r, target_eval_r, swapped, additional_samples, prev_r, prev_std = evaluate_policy(policy,
                                                                                           memory=replay_buffer,
                                                                                           force_swap=False,
                                                                                           cpg=args.cpg,
                                                                                           criterion=criterion,
                                                                                           eval_episodes=args.eval_episodes,
                                                                                           significance=args.significance,
                                                                                           prev_r=prev_r,
                                                                                           prev_std=prev_std,
                                                                                           reevaluate=args.reevaluate,
                                                                                           swap='swap' in args.policy_name)
    total_timesteps_with_eval += additional_samples
    results_dict['eval_rewards'].append((total_timesteps, eval_r))
    results_dict['target_eval_rewards'].append((total_timesteps, target_eval_r))
    results_dict['additional_samples'].append((total_timesteps, additional_samples))

    policy.save('model', directory=base_dir)
    with open(base_dir + '/results', 'wb') as f:
        pickle.dump(results_dict, f)
    env.close()
    eval_env.close()
