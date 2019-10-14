# Reproducibility Reinfcorement Learning

### Usage

```bash
# by pip
pip install --user -r requirements.txt

# by conda
conda env create -f environment.yml
conda activate rl2019

pytest
python run.py
python run.py --env_id=MsPacman-v0 --seed=11
python run_envs.py --num_seeds=5 --env_ids CartPole-v0 MsPacman-v0

tensorboard --logdir ./runs/
# http://localhost:6006
```

### Blog Post

https://morris-frank.github.io/uva-rl-reproducibility/
