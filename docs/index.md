# Semi-gradient vs full-gradient in Deep Q-learning

In this blog post we want to investigate the effectiveness os using the semi-gradient and the full-gradient in deep Q-learning in different enviroments.

We compared two different reinforcement learning methods in 3 environments.
Gradient methods have a deep policy which is updated by taking the gradient of the loss function over the weights.
The full-gradient method is unbiased since it relies only on the real target.
When using Monte-Carlo we aid our prediction with estimates. Due to these estimats our target is not unbiased anymore, due to that we take the semi-gradient.

For policy we used an epsilon gready policy ...
Q-learning value funtion

Implemented n-step for 0, 3, 8 and infinite steps.

Experiments of performance of gradient and semi-gradient on e3 environments from the gym library.
We chose the CartPole as representive of discrete environment.
Mountain Car for continious environment.
PacMan as more complex environment with a pixel observation space.

Each experiment ran for 100 episodes.


PacMan:
ConvNet architecture?
Training on GPU neccessary.

```bash
python run.py --env=MsPacman-v0 --seed=11
```

Cart Pole:

Mountain Car:


## Background

## Implementation

## Results

{% include CartPole-v0_duration.html %}
