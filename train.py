# coding:utf-8
__author__ = 'frkhit'

import sys

from demo import build_complex_model, build_model, LinearFit

if __name__ == '__main__':
    command = "train" if len(sys.argv) < 2 else sys.argv[1]
    if command == "simple":
        # build model for PAI-EAS
        print("building simple model...")
        build_model(_export_dir="./demo_simple")
    elif command == "complex":
        # build complex model for PAI-EAS
        print("building complex model...")
        build_complex_model(_export_dir="./demo_complex")
    else:
        # train model
        print("training...")
        LinearFit().train()
