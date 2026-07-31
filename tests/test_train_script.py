from scripts.train import get_parser


def test_train_script_parses_wandb_options():
    args = get_parser().parse_args(
        [
            "--K",
            "10",
            "--num_iterations",
            "20",
            "--spatial_preiterations",
            "5",
            "--torch_device",
            "cuda",
            "--initial_device",
            "cpu",
            "--dataset_path",
            "input.h5ad",
            "--output_path",
            "output.h5ad",
            "--use-wandb",
            "--wandb-project",
            "revisions",
            "--wandb-group",
            "simulation",
        ],
    )

    assert args.use_wandb
    assert args.wandb_project == "revisions"
    assert args.wandb_group == "simulation"
    assert args.spatial_preiterations == 5
