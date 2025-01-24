export PYTHONPATH=.
export WANDB_API_KEY="<your wandb api key goes here but not required for offline mode>"
python algo/cassiepede/evaluation.py \
  --seed 0 \
  --time_horizon 5000000 \
  --project_name roadrunner_cassiepede \
  --num_cassie_prob 0 0 0 1 \
  --position_offset 1.0 \
  --poi_heading_range 1.05 \
  --poi_position_offset 1.5 \
  --reward_name locomotion_cassiepede_feetairtime_modified \
  --perturbation_force 0.0 \
  --perturbation_torque 0.0 \
  --force_prob 0.0 \
  --runs_name "2024-08-12 00:13:36.585606" \
  --model_checkpoint latest \
  --evaluation_mode "interactive" \
  --state_history_size 1 \
  --terrain 1 \
  --offline \

