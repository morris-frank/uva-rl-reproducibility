# Semi-gradient vs full-gradient in Deep TD-learning

In this blog post we want to investigate the effectiveness of using the semi-gradient and the full-gradient updates for a deep value function approximator in Temporal-difference learning in different environments.


## Background
In Reinforcement Learning the general goal is to learn the best possible behavior, called _policy_ $$\pi$$, for a computer program, called the _agent_, given its current _state_ and the available _actions_. An environment could be a 2D maze, a [boardgame](https://nl.wikipedia.org/wiki/Backgammon), a [really complicated boardgame](https://deepmind.com/research/case-studies/alphago-the-story-so-far) or anything else what is learnable. Each environment gives rewards that can be negative or positive depending on what our agent does. The best possible policy maximizes these rewards. For example in the simple 2D maze, there is probably a big positive reward for reaching the end but maybe a small negative reward for every step taken, as walking is painful! This would encourage agents to find the exit in the shortest ammount of time.

<img src="https://imgs.xkcd.com/comics/computers_vs_humans.png" title="It's hard to train deep learning algorithms when most of the positive feedback they get is sarcastic.">

In the most naïve approach of Reinforcement Learning we would simply keep a table handy with all states and actions and just keep track of how much reward we have gotten subsequently from that position. During training we fill the table with the experience of our agent. Later the agent just needs to pick the most rewarding action from this table and take the next step.

### Value approximation
When number of states get too big to keep track of in a table, instead we can replace the table with function that approximates the correct table. This function, given a state, returns estimates of the corresponding values for all possible actions. We need to pick a family of functions that is capable of this complexity. In this research we focus on deep neural networks, as in [theory they can approximate any parametric function](https://en.wikipedia.org/wiki/Universal_approximation_theorem).

For training of our state-action value function we need a definition of the loss $$\mathcal{L}$$. The loss function tells the network how good the prediction (in our case the value of a action given the state) really is. If the loss gets small, the estimations given by our value function get closer to the truth. To find this loss function we do a little excursion to the Bellman equations.

Almost all of reinforcement learning is base on the [Bellman equations](https://joshgreaves.com/reinforcement-learning/understanding-rl-the-bellman-equations/). The most basic Bellman equation for our state-action value (called the $$Q$$-value) is:

$$Q_{\pi}(s, a) = \mathbb{E}_{s'}[r + \gamma\cdot \max_{a'}Q_\pi(s', a')]$$

This gives us the $$Q$$-value for state $$s$$ and action $$a$$ under the policy $$\pi$$. Now the expectation $$\mathbb{E}_{s'}$$ just means that given  we took action $$a$$ from state $$s$$ we might end up in different new states, as there is often randomness involved in these environments. The expectation gives us the weighted sum of all the following states. Inside for each of those we get the immediate reward $$r$$ (which might be negative!) times the maximal state-action value of this new state. The $$\gamma$$ is the _discount factor_, reduces the future reward. By having a $$\gamma < 1$$ we imply that we value direct rewards more than rewards in the far future. This is a common idea which can be found in economics and psychology as well.

The basic Bellman equation implies that we can learn $$Q(s,a)$$ by learning the the update of going from $$(s,a)$$ to $$(s',a')$$ which is one step in the future (the next action taken, afterwards). Therefore this method is called Temporal difference. We can extend the Temporal difference into a longer temporal chain, comparing the value $$(s,a)$$ the value of the state 2,3,… time-steps away. This yields the Bellman equation for _n_-step temporal differences:

$$
\begin{align*}
Q_\pi(s_t, a_t) &= \mathbb{E}_\tau [G + \gamma^{n+1} \cdot q(s_{t+n}, a_{t+n})]\\
G &= \sum_{i=0}^{n} \gamma^i \cdot r_{t+i}
\end{align*}
$$

As we're _n_ time-steps away from the state-action pair $$(s,a)$$ we need the expectation over all _trajectories_ $$\tau$$ to the target  pair $$(s_{t+n}, a_{t+n})$$. $$G$$ here is the _discounted_  reward, where we recursively apply the discount factor $$\gamma$$ to all the intermediate returns $$r_i$$ for the state transitions between $$s$$ and $$s_{t+n}$$.

A bigger $$n$$ reduces bias of the convergence, as we use more actual rewards, and reduces variance, due to the expectation over possible trajectories requiring less samples to converge. Note that if $$n$$ is equal to infinity, then we always reach the end of the episode, and we have a so called Monte Carlo method.

Although we don't have any convergence guarantees for the function approximation case, we can still make use of them for a loss function $$\mathcal{L}$$:

$$
\begin{align*}
\mathcal{L} &= \sum_s \mu(s) |Q_\pi(s,a) - \hat{Q}(s,a|w))|\\
Q_\pi(s,a) &= \mathbb{E}_\tau [G + \gamma^{n+1} Q_{\pi}(s_{t+n}, a_{t+n})]
\end{align*}
$$

<!--
PLEASE CITE BISHOP PAGE 199!
-->

$$v_\pi(s)$$ and $$\hat{v}_\pi(s, w)$$ are respectively the true value of $$s$$ under policy $$\pi$$ and the predicted or approximated value of $$s$$ under the parametrization w. $$\mu(s)$$ is the importance given to state s and normally approximated with the relative number of times it appears in the experiences we have with the environment.

Due to the usual impossibility of finding a closed-form solution to the minimization of $$\mathcal{L}$$, we turn to gradient-based methods, and specifically to [stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) (SGD).

With stochastic gradient descent, we try to iteratively decrease the loss by moving the parameters in the direction opposite of its gradient. The gradient is stochastic because we don't calculate the full gradient of all the states, instead calculating it only with the states we visited during the experiences. This will also remove the need to calculate $$\mu(s)$$ explicitly, as the SGD will update the states precisely in the proportion we visit them which is $$\mu(s)$$.

We can see that the gradient of our loss $$\mathcal{L}$$ with respect to the parameters $$w$$ of our network is:

$$
\frac{\partial}{\partial w} \mathcal{L} =
\begin{cases}
-\nabla \hat{Q}(s,a,w)), &Q_\pi(s,a) - \hat{Q}(s,a,w)) <= 0\\
\nabla \hat{Q}(s,a,w)), &Q_\pi(s,a) - \hat{Q}(s,a,w)) >= 0
\end{cases}
$$

We assume that the target $$Q_\pi(s,a)$$ is independent of the weights for our network $$w$$, which is not true, _as unless we reach the final state_, we still have to calculate the Q-value of the final state-action using $$w$$. Because of that, this gradient is called semi-gradient.
Without this assumption we would calculate the full gradient.

### Best practices
#### Experience Replay
Experience replay is a best practice used to smooth the variance of the weight updates to our network. For this we save the trajectories (the experience) in fixed size list and in every step we randomly sample a batch of experiences from this list to update with. More importantly this also breaks the temporal correlation between our samples, which is important as SGD needs independent samples. Experience replay reduces parameters' oscillation or likelihood to divergence (see the [DQN paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) for a complete justification).

#### $$\epsilon$$-decay
$$\epsilon$$-decay is another commonly used best practice. It employs an $$\epsilon$$-greedy strategy, but with a linear decay of its parameter $$\epsilon$$. This decrease is stopped when $$\epsilon$$ reaches some minimal value such as $$0.05$$. This allows for more _exploration_ in the beginning (actually complete randomness in the absolute beginning), which is necessary in the case of scarce rewards, and more _exploitation_ later on, allowing for better convergence of the state action values using a closer-to-optimal policy.


## Implementation
We want to experiment with the differences in using semi- vs full-gradient in Deep TD-learning. For this we build a simple neural network in PyTorch as our value function approximator. We always use the same network architecture as we just need a network which is capable enough to represent our value functions. As the the environments we are going to use are all rather simple we stick to the same network layout for simplicity. Input is the state description. If the state space is discrete, we use one-hot-encoding. Then we have one linear layer with 128 hidden units, a ReLU activation, and an output layer with one neuron for each action. For optimization We use the [Adam optimizer](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/).

For each experiment we have to set a learning rate $$\alpha$$, a discount factor $$\gamma$$, and the speed of the $$\epsilon$$ decay.
The memory for the experience replay is always fixed to 10k elements and a batch size of 64 is used for the SGD.

We built all this on top of the amazing deep learning library PyTorch.


## Environments
We conduct multiple experiments using different environments. We checked all of the 747 environments, at version `0.15.3`, of the OpenAI Gym package. The analysis on environment properties can be found [in a notebook](https://colab.research.google.com/drive/1ZAs_M0-0hrqrf9Qo7jkfJDrErRThpngZ), which lists environments sorted by complexity as measured by multiplied size of action and observation spaces, from easy to hard.

## Experiments
We test the difference between semi- and full-gradient in several different environments: [FrozenLake](https://gym.openai.com/envs/FrozenLake-v0/), [CartPole](https://gym.openai.com/envs/CartPole-v1/), [Acrobot](https://gym.openai.com/envs/Acrobot-v1/) and the [algorithmic environments](https://gym.openai.com/envs/#algorithmic).

### FrozenLake
[The FrozenLake](https://gym.openai.com/envs/FrozenLake-v0/) is a variant on the simple GridWorld. We have a small discrete 2D grid on which the actor can move in any of the four directions. The goal is to just go from one point to another point on the grid. But as the lake is frozen the agent might slip, so given an action the transition to another state is stochastic. Also the lake has ice-holes that, when fallen-in gives high negative reward. FrozenLake is an environment where both state and observation spaces are discrete, making it relatively simple compared to our other environments.

We set the learning rate $$\alpha=1e-5$$, train for 1.5k steps and decay $$\epsilon$$ to $$0.05$$ in 5k steps.
We run this experiment for 1-step TD (a.k.a. TD(0)) 10 times.

To replicate these exact experiments deterministically (using the same seeds for the pseudo-generator) run:
```bash
python run_envs.py --env_ids=FrozenLake-v0 --num_seeds=10 --it_at_min=5000 --alpha=1e-5 --n_episodes=1500 --n_step 0
```

Below we plot the average complete return (sum of the un-discounted rewards) of each episode over training as well as one standard deviation.
The runs using semi-gradient and using full-gradient are color-coded.

<figure>
{% include FrozenLake-v0_G.html %}
</figure>


### Cart Pole

<video autoplay loop controls>
    <source src="cartpole.mp4" type="video/mp4">
</video>

The next experiment is using the [CartPole-v0](https://gym.openai.com/envs/CartPole-v0/) environment, featuring a continuous observation space and a discrete action space.
In CartPole we have a cart with a pole on it. The state describes where the cart is, the angle of the pole and the velocity of cart and pole. The goal is to keep the pole upright for the whole episode. The episode is ended if the angle is too big (meaning the pole fell). The action we can take is putting force on the cart, either from the left or the right in each time-step.

As we want to compare the influence of the semi-gradient at different lengths of the dependency list for Q-learning, we test with $$n$$ steps, for $$n\in [0, 3, 8]$$.
$$\gamma$$ is fixed to $$0.8$$.
The learning rate $$\alpha$$ is fixed to $$1e-3$$.
We train for each n-step for 100 episodes and repeat each run five times.

To replicate this exact experiment run:
```bash
python run_envs.py --num_seeds=5 --alpha=1e-3 --gamma=0.8 --n_episodes=100 --n_step=1 --env_ids CartPole-v0
python run_envs.py --num_seeds=5 --alpha=1e-3 --gamma=0.8 --n_episodes=100 --n_step=4 --env_ids CartPole-v0
python run_envs.py --num_seeds=5 --alpha=1e-3 --gamma=0.8 --n_episodes=100 --n_step=8 --env_ids CartPole-v0
```

Below we plot the average duration of each episode over training as well as one standard deviation.
The runs using semi-gradient and using full-gradient are color-coded.

<figure>
{% include CartPole-v0_duration.html %}
</figure>


### Acrobot

<video autoplay loop controls>
    <source src="acrobot.mp4" type="video/mp4">
</video>

Next up we run experiments with the [Acrobot-v1](https://gym.openai.com/envs/Acrobot-v1/) environment from the OpenAI gym. In this environment we have a double pendulum with the goal to swing it high enough for the outer end to reach a certain height. The pendulum has two joints but only the lower one, between the two sticks, is moveable. As such as actions are putting torque on the joint in either direction or do nothing. So like CartPole, this game has a continuous observation space and a discrete action space.

For our experiment we set the learning rate $$\alpha$$ to $$1e-10$$, the discount factor $$\gamma$$ to $$0.97$$ and decay $$\epsilon$$ to $$0.05$$ it 1e5 steps. For each semi- and full-gradient we run 5 times for 1500 episodes.

As in all the experiments we have problems getting the model to consistently learn in a short amount of time. Out of five runs, for both semi- and full-gradient four did diverge and did not produce anything. Therefore we only plot the 1 run of each which did not diverge. Therefore we do only have one run to compare. Hyperparamter selection in RL is difficult.

To replicate these exact experiments run:
```bash
python run_envs.py --num_seeds=5 --alpha=1e-10 --gamma=0.97 --n_episodes=1500 --it_at_min=100000 --n_step=0 --env_ids Acrobot-v0
```

Below we plot the average duration of each episode over training as well as one standard deviation.
The runs using semi-gradient and using full-gradient are color-coded.

<figure>
{% include Acrobot-v1_G.html %}
</figure>

### MsPacman

<video autoplay loop controls>
    <source src="MsPacman.mp4" type="video/mp4">
</video>

Out of couriosity we tested our algorithm on the MsPacman game. This environment gives out the screen-pixels as observation space. There fore a convolutional neural net had to be implemented. We chose a simple architecture of two convolutional layers with batch normalization and max-pooling and two fully connected layers with a ReLU activation function.
The trainng could not be completed due to lacking computing power. None the less we are proud to present our self-learned Pacman agent after 30 hours of GPU trainng on Google colab. This game has been considered challenging by experts of the Googles AI research department [DeepMind](https://deepmind.com/).
<br/>










<br/>
### Algorithmic environments
The algorithmic environments are somewhat simpler, and similar in nature.
Like FrozenLake they feature discrete action and observation spaces.

We set the discount factor $$\gamma$$ to $$0.95$$. The learning rate is set to $$\alpha=1e-5$$.
We run the experiments for 4-step TD, 10 times.

To replicate these exact experiments run:
```bash
python run_envs.py --num_seeds=10 --alpha=1e-5 --gamma=0.95 --n_episodes=2000 --n_step=0 --env_ids Copy-v0 RepeatCopy-v0 Reverse-v0 DuplicatedInput-v0 ReversedAddition-v0 ReversedAddition3-v0
python run_envs.py --num_seeds=10 --alpha=1e-5 --gamma=0.95 --n_episodes=2000 --n_step=3 --env_ids Copy-v0 RepeatCopy-v0 Reverse-v0 DuplicatedInput-v0 ReversedAddition-v0 ReversedAddition3-v0
```

Below we plot the average return of each episode over training as well as one standard deviation.
The runs using semi-gradient and using full-gradient are color-coded.

#### [Copy](https://gym.openai.com/envs/Copy-v0/)
Input here is a random string, goal is to just produce the same string.

<figure>
{% include Copy-v0_G.html %}
</figure>

#### [DuplicatedInput](https://gym.openai.com/envs/DuplicatedInput-v0/)
Input here is a string of random chars which each are double. Goal is to have the chars only once, basically get every second character.

<figure>
{% include DuplicatedInput-v0_G.html %}
</figure>

#### [RepeatCopy](https://gym.openai.com/envs/RepeatCopy-v0/)
Input here is a random string and a integer m and goal is to produce a string of m concatenations of the input string.

<figure>
{% include RepeatCopy-v0_G.html %}
</figure>

#### [Reverse](https://gym.openai.com/envs/Reverse-v0/)
Input here is a random string and the goal is to reverse the string.

<figure>
{% include Reverse-v0_G.html %}
</figure>

## Discussion

The results did not show a clear advantage of using full-gradient over semi-gradient. While the computation of the full-gradient is only $O(1)$ with pytorch autograd, it did not outperform semi-gradient.
This might be due two reasons.

- First being the time spent training on each environment. Most games only reach the final reward after many steps. To avoid infinite episodes, a maximum has been set to terminate such episodes. This leads to the agent having few episodes where it reaches the final reward, and in the end it does not learn how to play the game in the allowed number of episodes. The solution to this is to increment the number of episodes.

- The second and more theoretical reason is that by using Q-learning, we already use a self-referential bootstrapping approach. Thus taking the full-gradient on the 'real' target is still an estimation biased just like the semi-gradient. This means that both methods use a very similar approach, which explains the similarity in our results.


## Conclusion
In this blog post we explored deep TD-learning and specifically the influence of using the full- or semi-gradient of the TD error as our weight update.

Unfortunatly the high computational cost of running reinforcement learning experiments in minimally complex environments limited the amount of experiences we could do. In these few runs the difference, in terms of speed of convergence or final results, had a too high variance to be possible to either prove or disprove it. Despite this it seems plausible that any difference is not enough to prefer one method over the other.

All code is available under [https://github.com/morris-frank/uva-rl-reproducibility](https://github.com/morris-frank/uva-rl-reproducibility).