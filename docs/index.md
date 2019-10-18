# Semi-gradient vs full-gradient in Deep TD-learning

In this blog post we want to investigate the effectiveness os using the semi-gradient and the full-gradient updates for a deep value function approximator in Temporal-difference learning in different environments.

## Background
In Reinforcement Learning the general goal is to learn for our  computer program (called the _agent_) what is best think to do given its current situation (called the _state_) given the available possible things to do (called _actions_). An environment can be anything like a 2D maze, a [boardgame](https://nl.wikipedia.org/wiki/Backgammon), a [really complicated boardgame](https://deepmind.com/research/case-studies/alphago-the-story-so-far) or anything else that's learnable. Each environment has rewards attach, that can be negative or positive depending on what our agent does. For example in  the simple 2D maze, there is probably a big positive reward for reaching the end but maybe a small negative reward for every step taken, as walking is painful!

In the most classical approach of Reinforcement Learning we would simply keep a table handy with all states and actions and just keep track of how much reward we have gotten subsequently from that position. During training we fill the table with the experience of our agent. Later the agent just needs to pick the most rewarding action from this table and that's it.

### Value approximation
When number of states get too big keep track of in a table, instead we can replace the table with function that approximates the correct table. This function, given a state, returns estimates of the corresponding values for all possible actions. We need to pick a family of functions that is capable of this complexity. In this research we focus on artificial neural networks, as in [theory they can approximate any parametric function](https://en.wikipedia.org/wiki/Universal_approximation_theorem).

For training of our state-action-value-function we need a definition of the loss. The loss function tells the network how good the prediction (in our case the value of a action given the state) really is. If the loss gets small, the estimations given by our value function get closer to the truth. Almost all of reinforcement learning is base on the [Bellman equations](https://joshgreaves.com/reinforcement-learning/understanding-rl-the-bellman-equations/). The most basic Bellman equation for our state-action value (called the $$Q$$-value) is:

$$Q_{\pi}(s, a) = E_{s'}[r + \gamma max_{a'}Q_\pi(s', a')]$$

Where $$Q_\pi(s, a)$$ is the true q value of all $$s, a$$ under a certain policy $$\pi$$, and $$\gamma$$ is a discount factor. We can use this formula to make updates to our state-action values, which in the tabular case is guaranteed to find the optimal value. This formula can be extended using a n-step temporal difference approach where we aim to have the following equality for all state-actions:

$$
Q_\pi(s_t, a_t) = E_\tau [G + \gamma^{n+1} q(s_{t+n}, a_{t+n})]\\
G = \sum_{i=0}^{n} \gamma^i r_{t+i}
$$

A bigger $$n$$ reduces bias of the convergence, as we use more actual rewards, and reduces variance, due to the expectation over possible trajectories requiring less samples to converge. Note that if $$n$$ is equal to infinity, then we always reach the end of the episode, and we have a Monte Carlo algorithm.

Although we don't have any convergence guarantees for the function approximation case, we can still make use of them for a loss function:

$$
L = \sum_s \mu(s) [q_\pi(s,a) - \hat{q}(s,a,w))]^2
q_(s,a) = E_\tau [G + \gamma^{n+1} q_pi(s_{t+n}, a_{t+n})]
$$

<!--
PLEASE CITE BISHOP PAGE 199!
-->

$$v_\pi(s)$$ and $$\hat{v}_\pi(s, w)$$ are respectively the true value of $$s$$ under policy $$\pi$$ and the predicted or approximated value of $$s$$ under the parametrization w. $$\mu(s)$$ is the importance given to state s and normally approximated with the relative number of times it appears in the experiences we have with the environment.

Due to the usual impossibility of finding a closed-form solution to the minimization of $$L$$, we turn to gradient-based methods, and specifically to [stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) (SGD).

With stochastic gradient descent, we try to iteratively decrease the loss by moving the parameters in the direction opposite of its gradient. It's stochastic because we don't calculate the full gradient of all the states, instead calculating it only with the states we visited during the experiences. This will also remove the need to calculate $$\mu(s)$$ explicitly, as it will more often update the states we visit the most in the correct proportion.

We can see that the gradient of our loss is:

$$
\frac{\partial}{\partial w} L = E_\tau[2 [q_\pi(s,a) - \hat{q}(s,a,w))] \nabla \hat{q}(s,a,w))]
$$

We assume that the target $$q_\pi(s,a)$$ is independent of the parametrization $$w$$, which is not true, as unless we reach the final state, we still have to calculate the q-value of the final state-action using $$w$$. Because of that, this gradient is called semi-gradient.
Without this assumption we would calculate the full gradient.

Experience replay is used as a mechanism that smooths the training distribution, and reduces correlation between used samples which is an assumption in stochastic gradient descent. It consists of sampling from a fixed size of the last experiences to update the network. This reduces parameters' oscillation or divergence (see [DQN paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)).

$$\epsilon$$-decay is also used as an exploration strategy. It employs an $$\epsilon$$-greedy strategy, but with a linear decay of its parameter. This decrease is stopped when $$\epsilon$$ reaches some minimal value such as 0.05. This allows for more exploration in the beginning, which is extremely necessary in the case of scarce rewards, and more exploitation later on, allowing for better convergence of the state action values using a closer-to-optimal policy.


## Environments

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

<!--
People also talk about like multi-player (co-op / adversarial) environments, but we see these just as regular environments with partial observability (gotta model the other actors, just as you'd model anything else in your environment). Maybe a difference is the other actors may also learn, invalidating your older intel. For our current purposes we will not focus on these environments though.
-->

Unfortunately, some properties are missing representation among Gym environments altogether:
- continuous time
- stateless games e.g. Multi-Armed Bandit

Our next step is to get a list of environments classified by these criteria.
[Environments](https://gym.openai.com/environments/Copy-v0/) we used all come directly from OpenAI's [gym](https://github.com/openai/gym) library, without additional extensions (e.g. OpenAI's [retro](https://github.com/openai/retro) environments).

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

For predicting the Q-values given a state, we need an approximation function.
For this, we use a simple neural network, consisting of a linear layer with 128 hidden units, a ReLU activation, and another linear layer.
We use the Adam optimizer, a smooth L1 loss, and a learning rate $$\alpha = 0.001$$.
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

### [Cart Pole](https://gym.openai.com/envs/CartPole-v1/)
<video autoplay loop controls>
    <source src="cartpole.mp4" type="video/mp4">
</video>

Our first experiment is using [OpenAI gym](http://gym.openai.com/)'s popular [CartPole-v0](https://gym.openai.com/environments/CartPole-v0/) enviroment, featuring a continuous observation space, yet a discrete action space.

As we want to compare the influence of the semi-gradient at different lengths of the dependency list for Q-learning, we test with $$n$$ steps, for $$n\in [0, 3, 8]$$.
$$\gamma$$ is fixed to $$0.8$$.
The learning rate for the approximation model $$\alpha$$ is fixed to $$1e-3$$.
We train for each n-step for 100 episodes and repeat each run five times.

Below we plot the average duration of each episode over training as well as one standard deviation.
The runs using semi-gradient and using full-gradient are color-coded.

<figure>
{% include CartPole-v0_duration.html %}
</figure>

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

The algorithmic environments are somewhat simpler, and similar in nature. Like FrozenLake and MsPacman, they feature discrete action and observation spaces.

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

#### [ReversedAddition](https://gym.openai.com/envs/ReversedAddition-v0/)

<figure>
{% include ReversedAddition-v0_G.html %}
</figure>
