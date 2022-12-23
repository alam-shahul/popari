import argparse

def main():
    parser = argparse.ArgumentParser(description='Run SpiceMix on specified dataset and device.')
    parser.add_argument('--path2dataset', type=str, help="Path to saved AnnData datasets.")
    parser.add_argument('--dataset_name', type=str, help="Dataset name.")
    parser.add_argument('--replicate_names', type=json.loads)
    parser.add_argument('--device', type=str, help="Device for PyTorch computation.")
    parser.add_argument('--random_seeds', type=str, help="Random seed range.")
    parser.add_argument('--differential', type=bool, help="Whether to do differential evaluation.")
    print('Running my program...')
