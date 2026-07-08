from popari.simulation.training import train_popari_with_warmup


class FakeParameterOptimizer:
    def __init__(self, calls):
        self.calls = calls

    def reinitialize_spatial_affinities(self):
        self.calls.append(("reinitialize_spatial_affinities",))


class FakeModel:
    def __init__(self):
        self.calls = []
        self.parameter_optimizer = FakeParameterOptimizer(self.calls)

    def estimate_parameters(self, **kwargs):
        self.calls.append(("estimate_parameters", kwargs))

    def estimate_weights(self, **kwargs):
        self.calls.append(("estimate_weights", kwargs))

    def synchronize_datasets(self):
        self.calls.append(("synchronize_datasets",))


def test_train_popari_with_warmup_runs_spatial_second_phase():
    model = FakeModel()

    returned = train_popari_with_warmup(
        model,
        nmf_iterations=2,
        spatial_iterations=1,
        verbose=0,
    )

    assert returned is model
    assert model.calls == [
        ("estimate_parameters", {"update_spatial_affinities": False}),
        ("estimate_weights", {"use_neighbors": False}),
        ("estimate_parameters", {"update_spatial_affinities": False}),
        ("estimate_weights", {"use_neighbors": False}),
        ("reinitialize_spatial_affinities",),
        ("synchronize_datasets",),
        ("estimate_parameters", {}),
        ("estimate_weights", {}),
    ]


def test_train_popari_with_warmup_can_keep_second_phase_nmf_like():
    model = FakeModel()

    train_popari_with_warmup(
        model,
        nmf_iterations=1,
        spatial_iterations=2,
        train_nmf_after_warmup=True,
        verbose=0,
    )

    assert model.calls == [
        ("estimate_parameters", {"update_spatial_affinities": False}),
        ("estimate_weights", {"use_neighbors": False}),
        ("reinitialize_spatial_affinities",),
        ("synchronize_datasets",),
        ("estimate_parameters", {"update_spatial_affinities": False}),
        ("estimate_weights", {"use_neighbors": False}),
        ("estimate_parameters", {"update_spatial_affinities": False}),
        ("estimate_weights", {"use_neighbors": False}),
    ]
