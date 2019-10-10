import torch
import numpy as np

class Approximator(torch.nn.Module):
    def __init__(self, net, alpha: float = 0.01, optimizer=torch.optim.Adam, loss=torch.nn.SmoothL1Loss):
        """

        :param net: The sequential network definition
        :param alpha: the learning rate
        :param optimizer: The optimizer to use
        :param loss:  the loss use for optimization
        """
        super(Approximator, self).__init__()
        self.net = net
        self.optimizer = optimizer(self.parameters(), lr=alpha)
        self.loss_function = loss()

    def forward(self, x):
        """
        Forward the state x
        :param x:
        :return:
        """
        return self.net(torch.FloatTensor(x))

    def batch_train(self, samples: list, gamma: float, semi_gradient: bool = True):
        """
        Train the network with the batch of experience samples
        :param samples: the list of samples, each sample is (G, start state, start action, last state, last action)
        :param gamma: the discount gamma for the last element in the trajectory (== gamma**n_step)
        :param semi_gradient: whether to use semi_gradient
        :return: the loss for the batch as float
        """
        # G, state τ, action τ, state t, action t
        Gs, τ_states, τ_actions, t_states, t_actions = zip(*samples)
        Gs = list(Gs)

        self.optimizer.zero_grad()
        if semi_gradient:
            torch.set_grad_enabled(False)

        # Compute the actual discounted returns:
        for i, (state, action) in enumerate(zip(t_states, t_actions)):
            if action is not None:
                Gs[i] = Gs[i] + gamma * self(state)[action].item()

        Gs = torch.tensor(Gs, dtype=torch.float)
        τ_states = torch.FloatTensor(τ_states)
        τ_actions = torch.tensor(τ_actions, dtype=torch.int64)

        torch.set_grad_enabled(True)
        target_q_vals = self(τ_states)
        target = target_q_vals[torch.arange(target_q_vals.size(0)), τ_actions]

        loss = self.loss_function(Gs, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()
