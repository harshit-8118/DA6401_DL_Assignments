import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
import wandb
from ann import NeuralNetwork
from utils import load_dataset, train_val_split, log_5_samples_from_each_class, optimizer_showdown, vanishing_grad_analysis
import argparse

CONFIG = {
    "beta": 0.9,
    "epsilon": 1e-8,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "nadam_beta1": 0.9,
    "nadam_beta2": 0.999,
    "val_split": 0.2,
    "train": False
}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a neural network")

    parser.add_argument(
        "-d", "--dataset", type=str, default="mnist", choices=["mnist", "fashion"]
    )
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("-b", "--batch_size", type=int, default=128)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    parser.add_argument(
        "-o",
        "--optimizer",
        type=str,
        default="adam",
        choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"],
    )
    parser.add_argument(
        "-l",
        "--loss",
        type=str,
        default="cross_entropy",
        choices=["mse", "cross_entropy"],
    )
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0)
    parser.add_argument("-nhl", "--hidden_layers", type=int, default=3)
    parser.add_argument(
        "-sz", "--num_neurons", type=int, nargs="+", default=[128, 64, 32]
    )
    parser.add_argument(
        "-a",
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "tanh", "sigmoid"],
    )
    parser.add_argument(
        "-wi", "--weight_init", type=str, default="xavier", choices=["random", "xavier"]
    )
    parser.add_argument("--wandb_project", type=str, default="DA6401_Assignment1")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="assignment-1\models")
    parser.add_argument("--model_save_path", type=str, default="best_model.npy")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")

    return parser.parse_args()

def main():
    args = parse_arguments()
    np.random.seed(args.seed)

    (X_train, Y_train), (x_test, y_test) = load_dataset(args.dataset)
    (x_train, y_train), (x_val, y_val) = train_val_split(
        X_train, Y_train, CONFIG["val_split"]
    )

    wandb_run = None
    if CONFIG["train"]:
        print("Training mode enabled. Starting training...")
        if not args.no_wandb:
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                config={
                    "dataset": args.dataset,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                    "optimizer": args.optimizer,
                    "loss": args.loss,
                    "weight_decay": args.weight_decay,
                    "num_neurons": args.num_neurons,
                    "activation": args.activation,
                    "weight_init": args.weight_init,
                },
                name=f"{args.optimizer}_lr{args.learning_rate}_{args.activation}",
            )

        model = NeuralNetwork(CONFIG, args)
        model.train(
            x_train,
            y_train,
            x_val,
            y_val,
            epochs=args.epochs,
            batch_size=args.batch_size,
            save_dir=args.save_dir,
            wandb_run=wandb_run,
        )

        test_m = model.evaluate(x_test, y_test, split_name="test")
        print(
            f"\nTest  ===== Loss: {test_m['loss']:.6f} | Acc: {test_m['accuracy']:.6f} | F1: {test_m['f1']:.6f}"
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "test/loss": test_m["loss"],
                    "test/accuracy": test_m["accuracy"],
                    "test/precision": test_m["precision"],
                    "test/recall": test_m["recall"],
                    "test/f1": test_m["f1"],
                }
            )
            wandb_run.finish()

        print("Training complete!")

    # if not args.no_wandb:
    #     if wandb_run is None:
    #         wandb_run = wandb.init(
    #             project=args.wandb_project,
    #             entity=args.wandb_entity,
    #             config={"dataset": args.dataset},
    #             name=f"Sample_Visualization_{args.dataset}",
    #         )
    #     log_5_samples_from_each_class(x_train, y_train, wandb_run=wandb_run)
    #     wandb_run.finish()
    # else:
    #     log_5_samples_from_each_class(x_train, y_train, wandb_run=None)

    # optimizer_showdown(args, CONFIG, x_train, y_train, x_val, y_val, NeuralNetwork=NeuralNetwork)    
    vanishing_grad_analysis(args, CONFIG, x_train, y_train, x_val, y_val, NeuralNetwork=NeuralNetwork)    

if __name__ == "__main__":
    main()

