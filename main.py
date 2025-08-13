import os
import sys
import argparse
from utils import import_custom_class


def main():

    parser = argparse.ArgumentParser(
        description="Arguments for the main train program."
    )
    parser.add_argument('--config_file', type=str, required=True, help='Path for the config file')
    parser.add_argument('--runner_class_path', type=str, default="runner/ge_trainer.py")
    parser.add_argument('--runner_class', type=str, default="Trainer")
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to trained checkpoint, used in inference stage only')
    parser.add_argument('--n_validation', type=int, default=1, help='num of samples to predict, used in inference stage only')
    parser.add_argument('--n_chunk_action', type=int, default=1, help='num of action chunks to predict, used in action inference stage only')
    parser.add_argument('--output_path', type=str, default=None, help='Path to save outputs, used in inference stage only')

    args = parser.parse_args()
    Runner = import_custom_class(
        args.runner_class, args.runner_class_path, 
    )
    

    if args.mode == "train":
        ### Trainer
        runner = Runner(args.config_file)
        runner.prepare_dataset()
        runner.prepare_models()
        runner.prepare_trainable_parameters()
        runner.prepare_optimizer()
        runner.prepare_for_training()
        runner.prepare_trackers()
        runner.train()

    elif args.mode == "infer":
        ### Inference
        runner = Runner(args.config_file, output_dir=args.output_path)
        if args.checkpoint_path is not None:
            runner.args.load_weights = True
            runner.args.load_diffusion_model_weights = True
            runner.args.diffusion_model['model_path'] = args.checkpoint_path
        runner.prepare_val_dataset()
        runner.prepare_models()
        runner.infer(
            n_chunk_action=args.n_chunk_action,
            n_validation=args.n_validation
        )

    else:
        raise NotImplementedError



if __name__ == "__main__":
  
    main()
    