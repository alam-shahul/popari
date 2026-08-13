import torch


class NesterovGD:
    """Optimizer implementing Nesterov accelerated gradient descent."""

    def __init__(self, parameters: torch.Tensor, step_size: float):
        self.parameters = parameters
        self.step_size = step_size
        self.y = torch.zeros_like(parameters)
        self.k = 0

    def set_parameters(self, parameters):
        """Reset the parameters updated by the optimizer."""

        self.parameters = parameters

    def step(self, grad: torch.Tensor):
        """Update parameters according to the current state and step size."""

        self.k += 1
        gamma = -(self.k - 1) / (self.k + 2)
        y_new = self.parameters - grad * self.step_size
        self.y = (self.y * gamma).add(y_new, alpha=(1 - gamma))
        self.parameters = self.y
        self.y = y_new
        return self.parameters
