# Semi-gradient vs full-gradient in Deep Q-learning

In this blog post we want to investigate the effectiveness os using the semi-gradient and the full-gradient in deep Q-learning in different enviroments.

## Background
Q-learning is Reinforcement learning method for learning an optimal policy in an given RL enviroment.
In Q-learning we want to find estimates for the Q-values.
The Q-values tell us given a state of our actor which action, if taken in this state will have the highest expected reward.

When state spaces get too big to represent in a tabular fashion it is natural to move into state value approximations where we try to create a function that, given the features of a state, returns a hopefully good guess of the corresponding value. Several possibilities exist for the construction of this function but we will focus on neural networks that can represent every possible parametric function.
In order to choose what is a good parametrization, the need for a loss function arises. The goal of function approximation will be therefore to minimize that loss. The mean Squared Value Error, denoted 
$$
\bar{VE}(w)
$$
is normally used.
$$
\bar{VE}(w) = \sum_{s\in S} \mu(s) \left[ v_\pi(s) - \hat{v}(s, w) \right]^2
$$
PLEASE CITE BISHOP PAGE 199!



v_\pi(s) and \hat{v}_\pi(s, w) are respectively the true value of s under policy \pi and the predicted or approximated value of s under the parametrization w. \mu(s) is the importance given to state s and normally approximated with the relative number of times it appears in the experiences we have with the environment. Due to the usual impossibility of finding a closed form solution to the minimization of \bar{VE}$$ we turn to gradient based methods specifically to stochastic gradient descent.

With stochastic gradient descent we try to iteratively decrease the loss by moving the parameters in the contrary direction of it's gradient. It's stochastic because we don't calculate the gradient of all the states instead calculating it only with the states we visited during the experiences. This will also remove the need to calculate $$mu(s)$$ explicitly as it will update more often the states we visit the most in the correct proportion. 

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
