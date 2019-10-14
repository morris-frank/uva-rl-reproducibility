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

## Background

## Implementation

## Results
We test our hypothesis in three different enviroments, the CartPole, the Pendulum and PacMan.

### Cart Pole
Our first experiment is using the [version 0 of the CartPole](https://gym.openai.com/envs/CartPole-v0/) enviroment from the OpenAI Gym.

As we want to compare the influence of the semi-gradient at different lengths of the dependency list for Q-learning we test with n-Steps: $$n\in [0, 3, 8]$$.
$$\gamma$$ is fixed to $$0.8$$.
The learning rate for the approximation model $$\alpha$$ is fixed to $$1e-3$$.
We train for each n-Step for 100 episodes and repeat each run five times.

Below we plot the average duration of each episode over training as well as one standard deviation.
The runs with using semi-gradient and using full-gradient are color-coded.

{% include CartPole-v0_duration.html %}


### Pendulum

### PacMan
PacMan:
ConvNet architecture?
Training on GPU neccessary.

```bash
python run.py --env=MsPacman-v0 --seed=11
```
