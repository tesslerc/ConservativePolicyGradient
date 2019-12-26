# ConservativePolicyGradient
Paper accompanying the IJCAI 2020 submission - Stabilizing Deep Reinforcement Learning with Conservative Updates

## Running the Conservative method:

python3.6 main.py --policy_name "TD3-swap" --env_name "Hopper-v2" --start_timesteps 1000 --swap_criterion "replay_prob" --experiment_name <name> --eval_freq 1000 --swap_start 0 --eval_episodes 1000 --significance 0.15 --target_distance_weight 0 --swap_freq <freq> --cpg

Or run the accompanying bash files.

Implementation based on the TD3 implementation by https://github.com/sfujim/TD3
