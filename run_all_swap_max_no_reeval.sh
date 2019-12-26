#!/bin/bash

for env in Hopper InvertedPendulum InvertedDoublePendulum
do
  for i in {0..4}
  do
    python3.6 main.py --policy_name "TD3-swap" --env_name "$env-v2" --start_timesteps 10000 --swap_criterion "eval_max" --experiment_name td3-prob-no-reeval-cpg --eval_freq 1000 --swap_start 0 --eval_episodes 10 --significance 0.15 --target_distance_weight 0 --no_reevaluation --swap_freq 10000 --cpg
  done
done

for env in Walker2d Ant HalfCheetah
do
  for i in {0..4}
  do
    python3.6 main.py --policy_name "TD3-swap" --env_name "$env-v2" --start_timesteps 10000 --swap_criterion "eval_max" --experiment_name td3-prob-no-reeval-cpg --eval_freq 1000 --swap_start 0 --eval_episodes 10 --significance 0.15 --target_distance_weight 0 --no_reevaluation --swap_freq 50000 --cpg
  done
done