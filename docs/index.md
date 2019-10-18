# Semi-gradient vs full-gradient in Deep TD-learning

In this blog post we want to investigate the effectiveness os using the semi-gradient and the full-gradient updates for a deep value function approximator in Temporal-difference learning in different environments.

## Background
In Reinforcement Learning the general goal is to learn for our computer program (called the _agent_) what is best think to do (called the _policy_ $$\pi$$) given its current situation (called the _state_) given the available possible things to do (called _actions_). An environment can be anything like a 2D maze, a [boardgame](https://nl.wikipedia.org/wiki/Backgammon), a [really complicated boardgame](https://deepmind.com/research/case-studies/alphago-the-story-so-far) or anything else that's learnable. Each environment has rewards attach, that can be negative or positive depending on what our agent does. For example in  the simple 2D maze, there is probably a big positive reward for reaching the end but maybe a small negative reward for every step taken, as walking is painful!

<img src="https://imgs.xkcd.com/comics/computers_vs_humans.png" title="It's hard to train deep learning algorithms when most of the positive feedback they get is sarcastic.">

In the most classical approach of Reinforcement Learning we would simply keep a table handy with all states and actions and just keep track of how much reward we have gotten subsequently from that position. During training we fill the table with the experience of our agent. Later the agent just needs to pick the most rewarding action from this table and that's it.

### Value approximation
When number of states get too big keep track of in a table, instead we can replace the table with function that approximates the correct table. This function, given a state, returns estimates of the corresponding values for all possible actions. We need to pick a family of functions that is capable of this complexity. In this research we focus on artificial neural networks, as in [theory they can approximate any parametric function](https://en.wikipedia.org/wiki/Universal_approximation_theorem).

For training of our state-action-value-function we need a definition of the loss $$\mathcal{L}$$. The loss function tells the network how good the prediction (in our case the value of a action given the state) really is. If the loss gets small, the estimations given by our value function get closer to the truth. To find this loss function we do a little excursion to the Bellman equations.

Almost all of reinforcement learning is base on the [Bellman equations](https://joshgreaves.com/reinforcement-learning/understanding-rl-the-bellman-equations/). The most basic Bellman equation for our state-action value (called the $$Q$$-value) is:

$$Q_{\pi}(s, a) = \mathbb{E}_{s'}[r + \gamma\cdot \max_{a'}Q_\pi(s', a')]$$

This gives us the $$Q$$-value for state $$s$$ and action $$a$$ under the policy $$\pi$$. Now the expectation $$\mathbb{E}_{s'}$$ just means that given  we took action $$a$$ from state $$s$$ we might end up in different new states, as there is often randomness involved in these environments. The expectation gives us the weighted sum of all the following states. Inside for each of those we get the immediate reward $$r$$ (which might be negative!) times the maximal state-action value of this new state. The $$\gamma$$ is the _discount factor_, reduces the future reward. By having a $$\gamma < 1$$ we imply that we value direct rewards more than rewards in the far future. This is common idea found in economics as well as psychology.

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
\mathcal{L} &= \sum_s \mu(s) [Q_\pi(s,a) - \hat{Q}(s,a|w))]^2\\
Q_(s,a) &= \mathbb{E}_\tau [G + \gamma^{n+1} Q_{\pi}(s_{t+n}, a_{t+n})]
\end{align*}
$$

<!--
PLEASE CITE BISHOP PAGE 199!
-->

$$v_\pi(s)$$ and $$\hat{v}_\pi(s, w)$$ are respectively the true value of $$s$$ under policy $$\pi$$ and the predicted or approximated value of $$s$$ under the parametrization w. $$\mu(s)$$ is the importance given to state s and normally approximated with the relative number of times it appears in the experiences we have with the environment.

Due to the usual impossibility of finding a closed-form solution to the minimization of $$L$$, we turn to gradient-based methods, and specifically to [stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) (SGD).

With stochastic gradient descent, we try to iteratively decrease the loss by moving the parameters in the direction opposite of its gradient. The gradient is stochastic because we don't calculate the full gradient of all the states, instead calculating it only with the states we visited during the experiences. This will also remove the need to calculate $$\mu(s)$$ explicitly, as the SGD will update the states precisely in the proportion we visit them which is $$\mu(s)$$.

We can see that the gradient of our loss $$\mathcal{L}$$ with respect to the parameters $$w$$ of our network is:

$$
\frac{\partial}{\partial w} \mathcal{L} = \mathbb{E}_\tau[2 [Q_\pi(s,a) - \hat{Q}(s,a,w))] \nabla \hat{Q}(s,a,w))]
$$

We assume that the target $$Q_\pi(s,a)$$ is independent of the weights for our network $$w$$, which is not true, _as unless we reach the final state_, we still have to calculate the Q-value of the final state-action using $$w$$. Because of that, this gradient is called semi-gradient.
Without this assumption we would calculate the full gradient.

### Best practices
#### Experience Replay
Experience replay is a best practice used to smooth the variance of the weight updates to our network. For this we save the trajectories (the experience) in fixed size list and in every step we randomly sample a batch of experiences from this list to update with. More importantly this also breaks the temporal correlation between our samples, which is important as SGD needs independent samples. Experience replay reduces parameters' oscillation or likelihood to divergence (see the [DQN paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) for a complete justification).

#### $$epsilon$$-decay
$$\epsilon$$-decay is another commonly used best practice. It employs an $$\epsilon$$-greedy strategy, but with a linear decay of its parameter $$\epsilon$$. This decrease is stopped when $$\epsilon$$ reaches some minimal value such as $$0.05$$. This allows for more _exploration_ in the beginning (actually complete randomness in the absolute beginning), which is necessary in the case of scarce rewards, and more _exploitation_ later on, allowing for better convergence of the state action values using a closer-to-optimal policy.


## Environments

We conduct multiple experiments using different environments. We went through all of the, at version `0.15.3`, 747 environments of the OpenAI Gym package. The analysis on environment properties can be found [in a notebook](https://colab.research.google.com/drive/1ZAs_M0-0hrqrf9Qo7jkfJDrErRThpngZ), which lists environments sorted by complexity as measured by multiplied size of action and observation spaces, from easy to hard.

We thought about different environment properties to compare by, and came up with the following:

state:
- stationary (`None`), discrete, continuous
- small vs large
- observable vs partially-observable
- deterministic vs stochastic (similar to above?)

action:
- discrete, continuous
- small vs large

reward:
- episodic vs continuing (discounting)

Unfortunately, some properties are missing representation among Gym environments altogether:
- continuous time
- stateless games e.g. Multi-Armed Bandit

Our next step is to get a list of environments classified by these criteria.
[Environments](https://gym.openai.com/environments/) we used all come directly from OpenAI's [gym](https://github.com/openai/gym) library, without additional extensions (e.g. OpenAI's [retro](https://github.com/openai/retro) environments).

Ideally, we want to be able to find environments with certain properties automatically. To do this, we want to be able to programmatically access the properties of the environment.

Unfortunately, some of these environment properties also cannot be distinguished programmatically using OpenAI Gym:
- episodic vs continuing tasks
- partial observability (unless you'd just consider all of Retro to qualify!)

For the other properties we found a way to list them from Gym to enable sorting/filtering.
This way we did some analysis on environment properties [in a notebook](https://colab.research.google.com/drive/1ZAs_M0-0hrqrf9Qo7jkfJDrErRThpngZ), which lists environments sorted by complexity as measured by multiplied size of action and observation spaces, from easy to hard.

From this, we have taken the following observations:
- 747 environments successfully load for us out of the box (which I think excludes Mujoco/Retro).
- most of these environments use discrete observation/action spaces and are non-stochastic. these properties somewhat limit their complexity, which is good news for us!
- the action space usually fits into 1 byte as well.
- the observation space can get either small (toy games) or large (roms played from pixels).
- roms usually have versions played on RAM as well, which use limited observation spaces and should be fully observable.
- ram versions are marked informally by their name -- if pixel roms were as easily identifiable we'd have our way to categorize roms by observability.
- some roms are also marked 'deterministic' by their name, yet their counterparts are also marked as deterministic in the metadata, so we do have some evidence the metadata is not fully reliable.
- environments by properties:
    - the only stochastic games use discrete act/obs spaces
    - both continuous: Pendulum / MountainCarContinuous
    - continuous state, discrete actions: has the infamous CartPole, plus a few others (easier/harder)
    - continuous actions, discrete observations: some options like Copy
    - stochastic, both spaces discrete: lists some versions of ElevatorAction-ram, which I don't buy as RAM should hold all state, so probably need some other stochastic env not formally listed as such (e.g. FrozenLake).
    - deterministic, both spaces discrete: this is the biggest category. This can just use a popular environment such as GridWorld.
    - stateless environments: multi-armed bandit? (not included in Gym)
    - big space: anything on pixels (too expensive?) e.g. CartPole

Based on this, we can probably just use some popular environments of manageable complexity, then maybe adjust/add as we learn more of how easy they are / how many we can take.

## Implementation

For this, we use a simple neural network, consisting of a linear layer with 128 hidden units, a ReLU activation, and another linear layer.
We use the [Adam optimizer](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/), a smooth L1 loss, and a learning rate $$\alpha = 0.001$$.
Our reinforcement learning agent consists of a semi-gradient Q-learning agent, with reward decay $$\gamma = 0.8$$, a final random action ratio $$\epsilon = 0.05$$ after 1,000 episodes of decay, a memory unit of 10,000 cells, 2 steps to consider, and a batch size of 64.

## Results

We test our hypothesis in a couple of different environments: [FrozenLake](https://gym.openai.com/envs/FrozenLake-v0/), [CartPole](https://gym.openai.com/envs/CartPole-v1/), [Acrobot](https://gym.openai.com/envs/Acrobot-v1/), [Ms PacMan](https://gym.openai.com/envs/MsPacman-v0/), and the [algorithmic environments](https://gym.openai.com/envs/#algorithmic).

The results did not show a clear advantage of using full-gradient over semi-gradient. While the computation of the full-gradient is only $O(1)$ more expensive, it did not outperform semi-gradient.
This might be due two reasons.
- First being the time spent training on each environment. Most games only reach the final reward after many steps. To avoid infinite episodes, a maximum has been set to terminate such episodes. This leads to the agent having few episodes where it reaches the final reward, and in the end it does not learn how to play the game. The solution to this is to increment the number of episodes, in the case of CartPole and Acrobot $100$ and $30$ for PacMan.
- The second and more theoretical reason is that by using Q-learning, we already use a self-referential bootstrapping approach. Thus taking the full-gradient on the 'real' target is still an estimation biased just like the semi-gradient. This means that both methods use a very similar approach, which explains the similarity in our results.

### FrozenLake

FrozenLake is an environment where both state and observation spaces are discrete, making it relatively simple compared to our other environments.

<figure>
{% include FrozenLake-v0_G.html %}
</figure>

To replicate these exact experiments deterministically (using the same seeds for the pseudo-generator) run:
```bash
python run_envs.py --num_seeds=10 --alpha=1e-5 --gamma=0.95 --n_episodes=2000 --n_step=3 --env_ids Copy-v0 RepeatCopy-v0 Reverse-v0 DuplicatedInput-v0 ReversedAddition-v0 ReversedAddition3-v0
```

### [Cart Pole](https://gym.openai.com/envs/CartPole-v1/)
<video autoplay loop controls>
    <source src="cartpole.mp4" type="video/mp4">
</video>

The next experiment is using the [CartPole-v0](https://gym.openai.com/environments/CartPole-v0/) environment, featuring a continuous observation space and a discrete action space.

As we want to compare the influence of the semi-gradient at different lengths of the dependency list for Q-learning, we test with $$n$$ steps, for $$n\in [0, 3, 8]$$.
$$\gamma$$ is fixed to $$0.8$$.
The learning rate for the approximation model $$\alpha$$ is fixed to $$1e-3$$.
We train for each n-step for 100 episodes and repeat each run five times.

Below we plot the average duration of each episode over training as well as one standard deviation.
The runs using semi-gradient and using full-gradient are color-coded.

<figure>
{% include CartPole-v0_duration.html %}
</figure>

To replicate this exact experiment run:
```bash
python run_envs.py --num_seeds=5 --alpha=1e-3 --gamma=0.8 --n_episodes=100 --n_step=1 --env_ids CartPole-v0
python run_envs.py --num_seeds=5 --alpha=1e-3 --gamma=0.8 --n_episodes=100 --n_step=4 --env_ids CartPole-v0
python run_envs.py --num_seeds=5 --alpha=1e-3 --gamma=0.8 --n_episodes=100 --n_step=8 --env_ids CartPole-v0
```


### [Acrobot](https://gym.openai.com/envs/Acrobot-v1/)

<video autoplay loop controls>
    <source src="acrobot.mp4" type="video/mp4">
</video>

Our second experiment concerns the [Acrobot-v1](https://gym.openai.com/environments/Acrobot-v1/) enviroment from the OpenAI gym. Like CartPole, this game has a continuous observation space and a discrete action space.

<figure>
{% include Acrobot-v1_G.html %}
</figure>

### [Ms PacMan](https://gym.openai.com/envs/MsPacMan-v0/)

<video autoplay loop controls>
    <source src="MsPacman.mp4" type="video/mp4">
</video>

<figure>
<!--
include MsPacman-v0_G.html
-->
</figure>

Ms PacMan is one of the Atari games contained in Gym. It has both a version played using RAM as input, as well as one using pixels as input. We used the latter here. This offers us an environment with discrete action and observation spaces, like FrozenLake, but characterized by a relatively large observation space compared to that of our other games.
To solve this game with the displayed pixels as observation space, a Convolutional Neural Network (CNN) is required. We opted for a simple architecture due to limited computational resources.
The CNN consists of the following (2d) layers:
- batch-norm
- convolution
- max-pool
- convolution
- max-pool
- linear
- ReLU
- linear

<!--
Training on GPU necessary.
-->

### [Algorithmic environments](https://gym.openai.com/envs/#algorithmic)

The algorithmic environments are somewhat simpler, and similar in nature.
Like FrozenLake and MsPacman, they feature discrete action and observation spaces.

#### [Copy](https://gym.openai.com/envs/Copy-v0/)

<figure>
{% include Copy-v0_G.html %}
</figure>

#### [DuplicatedInput](https://gym.openai.com/envs/DuplicatedInput-v0/)

<figure>
{% include DuplicatedInput-v0_G.html %}
</figure>

#### [RepeatCopy](https://gym.openai.com/envs/RepeatCopy-v0/)

<figure>
{% include RepeatCopy-v0_G.html %}
</figure>

#### [Reverse](https://gym.openai.com/envs/Reverse-v0/)

<figure>
{% include Reverse-v0_G.html %}
</figure>

To replicate these exact experiments run:
```bash
python run_envs.py --num_seeds=10 --alpha=1e-5 --gamma=0.95 --n_episodes=2000 --n_step=3 --env_ids Copy-v0 RepeatCopy-v0 Reverse-v0 DuplicatedInput-v0 ReversedAddition-v0 ReversedAddition3-v0
```
