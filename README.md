# MoVie: Visual Model-Based Policy Adaptation for View Generalization
Original PyTorch implementation of **MoVie** from

[MoVie: Visual Model-Based Policy Adaptation for View Generalization](https://yangsizhe.github.io/MoVie/) by

[Sizhe Yang](https://yangsizhe.github.io/)\*,   [Yanjie Ze](https://yanjieze.com/)\*,   [Huazhe Xu](http://hxu.rocks/)

<p align="center">
  <br><img src='media/overview.png' width="700"/><br>
</p>

## Method
MoVie is an effective approach to enable successful adaptation of visual model-based policies for view generalization during test time, without any need for reward signals and any modification during training time.

## Instructions
Assuming that you already have [MuJoCo](http://www.mujoco.org) installed, install dependencies using `conda`:

```
conda env create -f environment.yaml
conda activate movie
```

Install Adroit:

add `src/algorithms/modem/modem/tasks/mj_envs` to your `PYTHONPATH ` 

Install xArm:

```
cd src/envs/xarm_env
pip install -e. 
cd ../../..
```

Install wrappers for view generalization:

```
cd src/envs/distracting_control_viewgen
pip install -e. 
cd ../../..
```




For modem:  
configure wandb, `project_dir` and `model_path` in `cfgs/config_adaptation.yaml`(used for test time adaptation) and configure wandb and your demonstration/logging directories in `cfgs/config.yaml`(used for training).  
supported tasks: adroit, metaworld  
train with `train.sh` and eval with `eval.sh`      

For tdmpc:  
configure wandb, `model_path` and `eval_dir` in `cfgs/default_adaptation.yaml`(used for test time adaptation), and configure wandb in `cfgs/default.yaml`(used for training).  
supported tasks: dmc  
train with `train.sh` and eval with `eval.sh`   

(I will upload model files later.)  

After installing dependencies, you can train an agent by using the provided script
```
bash scripts/train.sh
```

## License & Acknowledgements
MoVie is licensed under the MIT license. MuJoCo is licensed under the Apache 2.0 license. 

We utilize the official implementation of TD-MPC and MoDem  which are available at \href{https://github.com/nicklashansen/tdmpc/}{github.com/nicklashansen/tdmpc} and \href{https://github.com/facebookresearch/modem/}{github.com/facebookresearch/modem} as the model-based reinforcement learning codebase. And the xArm environment is taken from:  
https://github.com/jangirrishabh/look-closer

