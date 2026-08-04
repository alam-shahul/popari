import torch
from torch import nn

from popari._named_state import ParameterDict


class SpatialAffinityState(nn.Module):
    """Registered spatial affinities and their runtime grouping behavior."""

    def __init__(self, K, sample_names, groups, mode, context):
        super().__init__()
        self.sample_names = tuple(sample_names)
        self.groups = {name: list(samples) for name, samples in groups.items()}
        self.mode = mode
        self.tags = {
            sample: [group for group, samples in self.groups.items() if sample in samples]
            for sample in self.sample_names
        }

        self.values = ParameterDict(prefix="spatial_affinity")
        if mode == "shared lookup":
            self._sample_to_parameter = {sample: group for group, samples in self.groups.items() for sample in samples}
            parameter_names = tuple(self.groups)
        elif mode == "differential lookup":
            self._sample_to_parameter = {sample: sample for sample in self.sample_names}
            parameter_names = self.sample_names
        else:
            raise NotImplementedError(f"{mode=} is not implemented.")
        self.parameter_names = tuple(parameter_names)
        for name in parameter_names:
            self.values[name] = torch.zeros((K, K), **context)

    def for_sample(self, sample: str) -> torch.Tensor:
        """Return the affinity matrix applicable to a sample."""

        return self.values[self._sample_to_parameter[sample]]

    def set_sample_(self, sample: str, value: torch.Tensor) -> None:
        """Copy an affinity matrix into the parameter applicable to a sample."""

        with torch.no_grad():
            self.for_sample(sample).copy_(value)

    def group_means(self) -> dict[str, torch.Tensor]:
        """Return detached arithmetic means of current sample affinities."""

        with torch.no_grad():
            return {
                group: torch.stack([self.for_sample(sample) for sample in samples]).mean(dim=0)
                for group, samples in self.groups.items()
            }

    def initialize_(self, sample_values: dict[str, torch.Tensor], betas: torch.Tensor) -> None:
        """Initialize state from one empirical affinity matrix per sample."""

        with torch.no_grad():
            if self.mode == "shared lookup":
                for group, samples in self.groups.items():
                    shared_affinity = self.values[group]
                    shared_affinity.zero_()
                    for sample, beta in zip(self.sample_names, betas):
                        if sample in samples:
                            shared_affinity.add_(beta * sample_values[sample])
            else:
                for sample in self.sample_names:
                    self.values[sample].copy_(sample_values[sample])

    def get_extra_state(self):
        return {
            "sample_names": self.sample_names,
            "groups": self.groups,
            "mode": self.mode,
        }

    def set_extra_state(self, state):
        if tuple(state["sample_names"]) != self.sample_names:
            raise RuntimeError(
                f"{self.__class__.__name__} checkpoint samples do not match the current samples.",
            )
        loaded_groups = {name: list(samples) for name, samples in state["groups"].items()}
        if loaded_groups != self.groups:
            raise RuntimeError(
                f"{self.__class__.__name__} checkpoint groups do not match the current groups.",
            )
        if state.get("mode") != self.mode:
            raise RuntimeError(f"{self.__class__.__name__} checkpoint mode is incompatible with the current model.")
