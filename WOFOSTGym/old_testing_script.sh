#!/bin/bash

# sleep until other tests have been run
#

# TEST 1 - run rppo and sac for 5M steps
# echo "beginning test 1..."
# nohup python -m train_agent --agent-type RPPO --env-id grape-lnpkw-v0 --agro-file grape_agro.yaml --save-folder data/grapevine/unconstrained_baseline_2/ --npk.seed 67 > data/grapevine/unconstrained_baseline_2/RPPO/output_67.log &
# nohup python -m train_agent --agent-type SAC --env-id grape-lnpkw-v0 --agro-file grape_agro.yaml --save-folder data/grapevine/unconstrained_baseline_2/ --npk.seed 67 > data/grapevine/unconstrained_baseline_2/SAC/output_67.log &

# TEST 2 - run maize baseline for ppo, rppo, sac, dqn
# echo "beginning test 2..."
# nohup python -m train_agent --agent-type DQN --env-id lnpkw-v0 --agro-file maize_agro.yaml --save-folder data/maize/unconstrained_baseline_1/ --npk.seed 67 > data/maize/unconstrained_baseline_1/DQN/output_67.log &
# nohup python -m train_agent --agent-type PPO --env-id lnpkw-v0 --agro-file maize_agro.yaml --save-folder data/maize/unconstrained_baseline_1/ --npk.seed 67 > data/maize/unconstrained_baseline_1/PPO/output_67.log &
# nohup python -m train_agent --agent-type RPPO --env-id lnpkw-v0 --agro-file maize_agro.yaml --save-folder data/maize/unconstrained_baseline_1/ --npk.seed 67 > data/maize/unconstrained_baseline_1/RPPO/output_67.log &
# nohup python -m train_agent --agent-type SAC --env-id lnpkw-v0 --agro-file maize_agro.yaml --save-folder data/maize/unconstrained_baseline_1/  --npk.seed 67 > data/maize/unconstrained_baseline_1/SAC/output_67.log &

# TEST 3 - run LAI_v1 reward test for PPO for 5M steps
# echo "beginning test 3..."
# python -m train_agent \
#     --agent-type PPO \
#     --env-id grape-lnpkw-v0 \
#     --agro-file grape_agro.yaml \
#     --save-folder data/grapevine/unconstrained_LAI_v1/ \
#     --env-reward RewardWrapper_LAI_v1 \
#     --npk.output-vars 'LAI', 'FIN', 'DVS', 'WSO', 'NAVAIL', 'PAVAIL', 'KAVAIL', 'SM', 'TOTN', 'TOTP', 'TOTK', 'TOTIRRIG' \
#     --npk.seed 67 > data/grapevine/unconstrained_LAI_v1/PPO/output_67.log 2>&1 &
# PID3=$!

# TEST 4 - run WSO_v1 reward test for PPO for 5M steps
# echo "beginning test 4..."
# python -m train_agent \
#     --agent-type PPO \
#     --env-id grape-lnpkw-v0 \
#     --agro-file grape_agro.yaml \
#     --save-folder data/grapevine/unconstrained_LAI_v2/ \
#     --env-reward RewardWrapper_LAI_v2 \
#     --npk.output-vars 'LAI', 'FIN', 'DVS', 'WSO', 'NAVAIL', 'PAVAIL', 'KAVAIL', 'SM', 'TOTN', 'TOTP', 'TOTK', 'TOTIRRIG' \
#     --npk.seed 67 > data/grapevine/unconstrained_LAI_v2/PPO/output_67.log 2>&1 &
# PID4=$!

# echo "beginning test 5 (LAI_v3)..."
# python -m train_agent \
#     --agent-type PPO \
#     --env-id grape-lnpkw-v0 \
#     --agro-file grape_agro.yaml \
#     --save-folder data/grapevine/unconstrained_LAI_v3/ \
#     --env-reward RewardWrapper_LAI_v3 \
#     --npk.output-vars LAI FIN DVS WSO NAVAIL PAVAIL KAVAIL SM TOTN TOTP TOTK TOTIRRIG \
#     --npk.seed 67 > data/grapevine/unconstrained_LAI_v3/PPO/output_67.log 2>&1 &
# PID5=$!


# echo "beginning test 6 (DVS_v1)..."
# python -m train_agent \
#     --agent-type PPO \
#     --env-id grape-lnpkw-v0 \
#     --agro-file grape_agro.yaml \
#     --save-folder data/grapevine/unconstrained_DVS_v1/ \
#     --env-reward RewardWrapper_DVS_v1 \
#     --npk.seed 67 > data/grapevine/unconstrained_DVS_v1/PPO/output_67.log 2>&1 &
# PID6=$!


# echo "beginning test 7 (TAGP_v1)..."
# python -m train_agent \
#     --agent-type PPO \
#     --env-id grape-lnpkw-v0 \
#     --agro-file grape_agro.yaml \
#     --save-folder data/grapevine/unconstrained_TAGP_v1/ \
#     --env-reward RewardWrapper_TAGP_v1 \
#     --npk.output-vars TAGP FIN DVS WSO NAVAIL PAVAIL KAVAIL SM TOTN TOTP TOTK TOTIRRIG \
#     --npk.seed 67 > data/grapevine/unconstrained_TAGP_v1/PPO/output_67.log 2>&1 &
# PID7=$!

# echo "beginning test 8 (DVR_v1)..."
# python -m train_agent \
#     --agent-type PPO \
#     --env-id grape-lnpkw-v0 \
#     --agro-file grape_agro.yaml \
#     --save-folder data/grapevine/unconstrained_DVR_v1/ \
#     --env-reward RewardWrapper_DVR_v1 \
#     --npk.output-vars DVR FIN DVS WSO NAVAIL PAVAIL KAVAIL SM TOTN TOTP TOTK TOTIRRIG \
#     --npk.seed 67 > data/grapevine/unconstrained_DVR_v1/PPO/output_67.log 2>&1 &
# PID8=$!

# echo "beginning test 9 (GRLV_v1)..."
# python -m train_agent \
#     --agent-type PPO \
#     --env-id grape-lnpkw-v0 \
#     --agro-file grape_agro.yaml \
#     --save-folder data/grapevine/unconstrained_GRLV_v1/ \
#     --env-reward RewardWrapper_GRLV_v1 \
#     --npk.output-vars GRLV FIN DVS WSO NAVAIL PAVAIL KAVAIL SM TOTN TOTP TOTK TOTIRRIG \
#     --npk.seed 67 > data/grapevine/unconstrained_GRLV_v1/PPO/output_67.log 2>&1 &
# PID9=$!

mkdir data/grapevine/unconstrained_LAI_v4/
mkdir data/grapevine/unconstrained_LAI_v4/PPO/
echo "beginning test 10 (LAI_v4)..."
python -m train_agent \
    --agent-type PPO \
    --env-id grape-lnpkw-v0 \
    --agro-file grape_agro.yaml \
    --save-folder data/grapevine/unconstrained_LAI_v4/ \
    --env-reward RewardWrapper_LAI_v4 \
    --npk.output-vars TAGP FIN LAI WSO NAVAIL PAVAIL KAVAIL SM TOTN TOTP TOTK TOTIRRIG \
    --npk.seed 67 > data/grapevine/unconstrained_LAI_v4/PPO/output_67.log 2>&1 &
PID10=$!

mkdir data/grapevine/unconstrained_LAI_v5/
mkdir data/grapevine/unconstrained_LAI_v5/PPO/
echo "beginning test 11 (LAI_v5)..."
python -m train_agent \
    --agent-type PPO \
    --env-id grape-lnpkw-v0 \
    --agro-file grape_agro.yaml \
    --save-folder data/grapevine/unconstrained_LAI_v5/ \
    --env-reward RewardWrapper_LAI_v5 \
    --npk.output-vars LAI FIN DVS WSO NAVAIL PAVAIL KAVAIL SM TOTN TOTP TOTK TOTIRRIG \
    --npk.seed 67 > data/grapevine/unconstrained_LAI_v5/PPO/output_67.log 2>&1 &
PID11=$!

# wait for processes to finish
echo "Waiting for tests to complete..."
wait $PID10 $PID11

echo "tests complete!"