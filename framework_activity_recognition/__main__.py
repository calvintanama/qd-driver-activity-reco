import argparse
from framework_activity_recognition.io import load_config_file
from framework_activity_recognition.driver import train, test_benchmark

def main():
    parser = argparse.ArgumentParser("Training framework")
    parser.add_argument("training_type", type=str, choices={"train", "test"}, help="choose training type")
    parser.add_argument("config_path", type=str, help="path to the yaml configuration file")
    parser.add_argument("--checkpoint", type=str, help="path to the .pt file to continue training.\
        This will be ignored if pretrained model is used")
    parser.add_argument("--output", type=str, help="path to save checkpoints")

    args = parser.parse_args()

    config = load_config_file(args.config_path)
    
    if args.training_type == "train":
        train(config)
    if args.training_type == "test":
        test_benchmark(config)


if __name__ == "__main__":
    main()