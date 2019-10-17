# Semi-gradient vs full-gradient in Deep Q-learning

In this blog post we want to investigate the effectiveness os using the semi-gradient and the full-gradient in deep Q-learning in different enviroments.

## Background
When state spaces get too big to represent in a tabular fashion it is natural to move into action state value approximations where we try to create a function that, given the features of a state, returns a hopefully good guess of the corresponding actions values. Several possibilities exist for the construction of this function but we will focus on neural networks that can represent every possible parametric function.
When trying to get a good parametrization the need for a loss function arises. The goal of function approximation will be therefore to minimize that loss. Here we turn to the Bellman equations and n-step TD learning to get a good loss function.

$$Q_{\pi}(s, a) = E_{s'}[r + \gamma max_{a'}Q_\pi(s', a')]$$

Where Q_\pi(s, a) is the true q value of all $$s a$$ under a certain policy $$\pi$$ and $$\gamma$$ is a discount factor. We can use this formula to make updates to our state action values which guarantees us and in the tabular case it this is guaranteed to find the optimal value. This formula can be extended using a n step temporal difference approach where we aim to have the following equality for all state actions:

$$
Q_\pi(s_t, a_t) = E_\tau [G + \gamma^{n+1} q(s_{t+n}, a_{t+n})]\\
G = \sum_{i=0}^{n} \gamma^i r_{t+i}
$$

A bigger n reduces bias of the convergence (as we use more actual rewards) and a reduces variance due to the expectation over possible trajectories requiring less samples to converge. We notice that if n is equal to infinity then we always reach the end of the episode and we have a Monte Carlo algorithm.

Although we don't have any convergence guarantees for the function approximation case we can still make use of them for a loss function:

$$
L = \sum_s \mu(s) [q_\pi(s,a) - \hat{q}(s,a,w))]^2
q_(s,a) = E_\tau [G + \gamma^{n+1} q_pi(s_{t+n}, a_{t+n})]
$$

PLEASE CITE BISHOP PAGE 199!

$$v_\pi(s)$$ and $$\hat{v}_\pi(s, w)$$ are respectively the true value of $$s$$ under policy $$\pi$$ and the predicted or approximated value of $$s$$ under the parametrization w. $$\mu(s)$$ is the importance given to state s and normally approximated with the relative number of times it appears in the experiences we have with the environment.

Due to the usual impossibility of finding a closed form solution to the minimization of $$L$$ we turn to gradient based methods specifically to stochastic gradient descent.

With stochastic gradient descent we try to iteratively decrease the loss by moving the parameters in the contrary direction of its gradient. It's stochastic because we don't calculate the full gradient of all the states, instead calculating it only with the states we visited during the experiences. This will also remove the need to calculate $$mu(s)$$ explicitly as it will update more often the states we visit the most in the correct proportion.

We can see that the gradient of our loss is

$$
\frac{\partial}{\partial w} L = E_\tau[2 [q_\pi(s,a) - \hat{q}(s,a,w))] \nabla \hat{q}(s,a,w))]
$$

We assume that the target $$q_\pi(s,a)$$ is independent of the parametrization $$w$$ which is not true as unless we reach the final state we still have to calculate the q value of the final state, action using $$w$$. Because of that this gradient is called semi-gradient.
Without this assumption we would calcutate the full gradient.

Experience replay is used as a mechanism that smooths the training distribution and reduces correlation between used samples which is a assumption in stochastic gradient descent. It consists of sampling from a fixed size of the last experiences to update the network. This reduces parameters oscilation or divergence. PLEASE CITE DQN PAPER.

$$\epsilon$$ decay is also used as an exploration strategy. It employs an $$\epsilon$$ greedy strategy but with a linear decay of its parameter. This decrease is stopped when $$\epsilon$$ reaches some minimal value such as 0.05. This allows for more exploration in the beginning, which is extremely necessairy in the case of scarce rewards, and more exploitation later on, allowing for better convergence of the state action values using a closer to optimal policy.


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

People also talk about like multi-player (co-op / adversarial) environments, but we see these just as regular environments with partial observability (gotta model the other actors, just as you'd model anything else in your environment). Maybe a difference is the other actors may also learn, invalidating your older intel. For our current purposes we will not focus on these environments though.

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
For prediciting the Q-values given a state we need an approximation function.

## Results
We test our hypothesis in three different enviroments, the CartPole, the Acrobot and PacMan.

The results did not show a clear advantage of using full-gradient over semi-gradient. While the computation of the full-gradient is only $O(1)$ more expensive it did not outperform semi-gradient.
This might be due two reasons. First being the time spend training on each environment. Most games only reach the final reward after many steps. To avoid infinite episodes a maximum has been set to terminate such episodes. This leads to that the agent has little episodes where he reaches the final reward and eventually does not learn how to play the game. The solution to this is to increment the number of episodes, in the case of CartPole and Acrobot $100$ and $30$ for PacMan.
The second and more theoretical reason is, that by using Q-learning we already use a self-referential bootstraping approach. Thus taking the full-gradient on the "real" Targed is still estiamtion biased as is the semi-gradient. This concludes that both methods use a very similar approach what explains the similarity in our results.

### FrozenLake

<figure>
{% include FrozenLake-v0_G.html %}
</figure>

### Cart Pole
<video autoplay loop controls>
    <source src="cartpole.mp4" type="video/mp4">
</video>

Our first experiment is using the [CartPole-v0](https://gym.openai.com/environments/CartPole-v0/) enviroment from the OpenAI gym.

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

Our second experiment concerns the [Acrobot-v1](https://gym.openai.com/environments/Acrobot-v1/) enviroment from the OpenAI gym.

<figure>
{% include Acrobot-v1_G.html %}
</figure>


### Algorithms
#### Copy
<figure>
{% include Copy-v0_G.html %}
</figure>

#### DuplicatedInput
<figure>
{% include DuplicatedInput-v0_G.html %}
</figure>

#### RepeatCopy
<figure>
{% include RepeatCopy-v0_G.html %}
</figure>

#### Reverse
<figure>
{% include Reverse-v0_G.html %}
</figure>

#### ReversedAddition
<figure>
{% include ReversedAddition-v0_G.html %}
</figure>

### PacMan
VIDEO
PLOTS

To solve the PacMan game with the displayed pixels as observation space a Convolutional Neural Network (CNN) is required. We opted for a simple architecture due to limited computational resources.
The CNN has
PacMan:
ConvNet architecture?
Training on GPU neccessary.
