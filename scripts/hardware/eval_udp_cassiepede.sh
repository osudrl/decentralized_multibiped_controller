export PYTHONPATH=.
export WANDB_API_KEY="<your wandb api key goes here>"
python cassiepede_udp.py \
  --state_dim 41 \
  --hidden_dim 64 \
  --lstm_hidden_dim 64 \
  --lstm_num_layers 2 \
  --set_adam_eps \
  --eps 1e-5 \
  --use_orthogonal_init \
  --seed 0 \
  --std 0.13 \
  --model_checkpoint latest \
  --project_name roadrunner_cassiepede \
  --reward_name locomotion_cassiepede \
  --reward_name locomotion_cassiepede_clock_stand \
  --reward_name locomotion_cassiepede_feetairtime_modified \
  --run_name "2024-05-01 10:30:51.809112" \
  --my_address "192.168.2.29" 1234 \
  --encoding 0.0 0.0 \
  --redownload_checkpoint \
  --do_log
