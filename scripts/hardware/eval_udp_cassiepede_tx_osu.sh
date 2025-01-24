export PYTHONPATH=.
export WANDB_API_KEY="<your wandb api key goes here>"
python cassiepede_tx_udp.py \
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
  --reward_name locomotion_cassiepede_feetairtime_modified \
  --encoding 0.0 0.0 \
  --transformer_hidden_dim 32 \
  --transformer_num_layers 2 \
  --transformer_num_heads 2 \
  --transformer_dim_feedforward 32 \
  --state_history_size 8 \
  --do_log \
  --run_name "2024-05-01 10:30:51.809112" \
  --redownload_checkpoint \
