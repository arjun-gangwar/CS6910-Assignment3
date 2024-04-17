import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

def main(args: argparse.Namespace):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="training file for assignment 3")
    parser.add_argument("--use_wandb")
    args = parser.parse_args()

    main(args)


