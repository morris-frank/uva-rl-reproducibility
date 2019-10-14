# Semi-gradient vs full-gradient in Deep Q-learning

In this blog post we want to investigate the effectiveness os using the semi-gradient and the full-gradient in deep Q-learning in different enviroments.

## Background
Q-learning is Reinforcement learning method for learning an optimal policy in an given RL enviroment.
In Q-learning we want to find estimates for the Q-values.
The Q-values tell us given a state of our actor which action, if taken in this state will have the highest expected reward.

## Implementation
For prediciting the Q-values given a state we need an approximation function.

## Results
We test our hypothesis in three different enviroments, the CartPole, the Acrobot and PacMan.

### Cart Pole
<video autoplay loop controls>
    <source src="cartpole.mp4" type="video/mp4">
</video>

Our first experiment is using the [CartPole-v0](https://gym.openai.com/envs/CartPole-v0/) enviroment from the OpenAI gym.

As we want to compare the influence of the semi-gradient at different lengths of the dependency list for Q-learning we test with n-Steps: $$n\in [0, 3, 8]$$.
$$\gamma$$ is fixed to $$0.8$$.
The learning rate for the approximation model $$\alpha$$ is fixed to $$1e-3$$.
We train for each n-Step for 100 episodes and repeat each run five times.

Below we plot the average duration of each episode over training as well as one standard deviation.
The runs with using semi-gradient and using full-gradient are color-coded.

<figure>
{% include CartPole-v0_duration.html %}
</figure>

### Acrobot
<video autoplay loop controls>
    <source src="acrobot.mp4" type="video/mp4">
</video>

Our second experiment concerns the [Acrobot-v1](https://gym.openai.com/envs/Acrobot-v1/) enviroment from the OpenAI gym.

### PacMan
PacMan:
ConvNet architecture?
Training on GPU neccessary.

```bash
python run.py --env_id=MsPacman-v0 --seed=11
```
