The code for paper: 《Tuning Synaptic Connections instead of Weights by Genetic Algorithm in Spiking Policy Network》


## Requirements

- Gym 0.15.7 
- mujoco-py 2.0.2.13
- Python 3.6.13
- PyTorch 1.6.0+cpu


The code has been tested on an AMD EPYC 7742 server without using GPU.
<br>

## Explanations
- ./results: the saved checkpoints and results on 10 runs 
- ./CartPole-v1, HalfCheetah-v2, Swimmer-v2, or HumanoidStandup-v2: training log
- result_analyze.py: result analysis and visualization code
- ./imgs: visualization results in our paper


## Run Steps

```bash
bash run.sh # Need to specify env_name(CartPole-v1, HalfCheetah-v2, Swimmer-v2, or HumanoidStandup-v2) and policy_type(SPN_Weights or SPN_Connections).
```