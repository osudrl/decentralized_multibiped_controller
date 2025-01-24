# Decentralized Multi-Biped Controller

This code is heavily adopted from https://github.com/osudrl/roadrunner

Website link: https://decmbc.github.io/
Paper link: https://arxiv.org/abs/2406.17279
Demo link: https://www.youtube.com/watch?v=2sJQCBaYKsw

### Steps to run the evaluation

1. To run the evaluation on the flat terrain with pretrained policy run
`./scripts/eval_cassiepede.sh`

2. To run the evaluation on the uneven terrain with pretrained policy run
`./scripts/eval_cassiepede_terrain.sh`

3. Change command line argument in the script as requires. For example supply different terrain by changing terrain index in `--terrain`

### Steps to run the training

1. To run the training
`./scripts/train_cassiepede.sh`
