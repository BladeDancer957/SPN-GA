#!/bin/bash

env_name='CartPole-v1'
policy_type='SPN_Connections'
popsize=200
generations=100

for seed in 10 20 30 40 50 60 70 80 90 100
do
  nohup python -u main.py --env_name=${env_name} --policy_type=${policy_type} --popsize=${popsize} --generations=${generations} --seed=${seed} >./${env_name}/${policy_type}_${generations}_${popsize}_${seed}.log 2>&1 &
done