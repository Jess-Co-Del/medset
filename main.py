"""
Segmentation app interface.
"""
import argparse
from app import app


if __name__ == "__main__":
    parse=argparse.ArgumentParser()
    parse = argparse.ArgumentParser()
    parse.add_argument("action", type=str, help="train or test")
    args = parse.parse_args()

    if args.action == "train":
        app = app.Application().model_trainer()
    elif args.action == "test":
        app = app.Application().model_inference()
    elif args.action == "cv":
        app = app.Application().model_cross_validate()
    elif args.action == "cv_inference":
        app = app.Application().model_cross_val_inference()
