export PYTHONPATH=.
export WANDB_API_KEY="<your wandb api key goes here>"
python cassiepede_udp.py \
  --hidden_dim 64 \
  --lstm_hidden_dim 64 \
  --lstm_num_layers 2 \
  --set_adam_eps \
  --eps 1e-5 \
  --use_orthogonal_init \
  --seed 0 \
  --std 0.13 \
  --model_checkpoint latest \
  --reward_name locomotion_cassiepede_feetairtime_modified \
  --run_name "2024-05-01 10:30:51.809112" \
  --encoding 0.7 0.0 \
  --do_log
