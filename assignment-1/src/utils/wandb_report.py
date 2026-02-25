import numpy as np
import wandb
import matplotlib.pyplot as plt
import argparse

def log_5_samples_from_each_class(X, y, image_shape=(28, 28), wandb_run=None):
    """
    Logs 5 raw samples per class to Weights & Biases as individual images,
    displayed row-wise (one row per class).

    Args:
        X (numpy.ndarray): Input images (n_samples, ...) — can be flat or 2D
        y (numpy.ndarray): One-hot encoded labels (n_samples, n_classes)
        image_shape (tuple): Shape to reshape flat images (default: 28x28 for MNIST)
        wandb_run: Active wandb run object
    """

    # --- Decode one-hot → class indices ---
    if y.ndim > 1:
        y = np.argmax(y, axis=-1)          # (n_samples, 10) → (n_samples,)

    classes = np.unique(y)                 # [0, 1, 2, ..., 9] for MNIST
    num_classes = len(classes)             # 10

    print(f"Unique classes found: {classes}")

    # --- Build grid: num_classes rows × 5 cols ---
    fig, axes = plt.subplots(
        num_classes, 5,
        figsize=(10, 2 * num_classes)      # 10 wide, 2px height per row → clean for 10 classes
    )

    # Safety: if somehow only 1 class, axes is 1D → make it 2D
    if num_classes == 1:
        axes = np.expand_dims(axes, axis=0)

    # --- Collect wandb images as list (one entry per image, captioned) ---
    wandb_images = []

    for i, cls in enumerate(classes):

        class_indices = np.where(y == cls)[0]
        num_samples = min(5, len(class_indices))

        selected_indices = np.random.choice(
            class_indices,
            size=num_samples,
            replace=False
        )

        for j in range(5):
            ax = axes[i, j]
            ax.axis("off")

            if j < num_samples:
                img = X[selected_indices[j]]

                # Reshape if flat (e.g., 784 → 28×28)
                if img.ndim == 1:
                    img = img.reshape(image_shape)

                # Normalize to [0, 255] uint8 if float
                if img.dtype != np.uint8:
                    img = (img * 255).clip(0, 255).astype(np.uint8)

                # Plot in grid
                axes[i, j].imshow(img, cmap="gray")
                axes[i, j].set_title(f"Class {cls}", fontsize=8)

                # Log each image individually to wandb with caption
                wandb_images.append(
                    wandb.Image(img, caption=f"Class {cls} | sample {j+1}")
                )

    plt.tight_layout()

    # Log individual raw images as a gallery
    if wandb_run is not None:
        wandb_run.log({"samples_per_class": wandb_images})

        # Also log the full grid as a single overview figure
        wandb_run.log({"samples_grid": wandb.Image(fig)})

    plt.show()
    plt.close(fig)


def optimizer_showdown(args, config, x_train, y_train, x_val, y_val, NeuralNetwork=None):
    """
    2.3 — Runs all 6 optimizers on fixed architecture and logs convergence curves.
    Fixed: 3 hidden layers, 128 neurons each, ReLU, same LR, same dataset.
    """

    OPTIMIZERS = ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]

    for opt in OPTIMIZERS:

        print(f"\n{'='*50}")
        print(f"  Optimizer Showdown: {opt.upper()}")
        print(f"{'='*50}")

        # Override only optimizer — keep everything else fixed
        showdown_args = argparse.Namespace(
            dataset       = args.dataset,
            epochs        = args.epochs,
            batch_size    = args.batch_size,
            learning_rate = args.learning_rate,
            optimizer     = opt,
            loss          = args.loss,
            weight_decay  = 0.0,
            num_neurons   = [128, 128, 128],   # fixed: 3 layers × 128
            activation    = "relu",             # fixed: ReLU
            weight_init   = args.weight_init,
            seed          = args.seed,
            save_dir      = args.save_dir,
            model_save_path = f"showdown_{opt}.npy",
        )

        wandb_run = None
        if not args.no_wandb:
            wandb_run = wandb.init(
                project = args.wandb_project,
                entity  = args.wandb_entity,
                config  = {
                    "experiment"   : "2.3_optimizer_showdown",
                    "optimizer"    : opt,
                    "epochs"       : showdown_args.epochs,
                    "batch_size"   : showdown_args.batch_size,
                    "learning_rate": showdown_args.learning_rate,
                    "num_neurons"  : showdown_args.num_neurons,
                    "activation"   : showdown_args.activation,
                    "dataset"      : showdown_args.dataset,
                },
                name   = f"showdown_{opt}",
                group  = "2.3_optimizer_showdown",   # groups all 6 runs in W&B
                reinit = True,
            )

        model = NeuralNetwork(config, showdown_args)
        model.train(
            x_train, y_train,
            x_val,   y_val,
            epochs     = showdown_args.epochs,
            batch_size = showdown_args.batch_size,
            save_dir   = showdown_args.save_dir,
            wandb_run  = wandb_run,
        )

        if wandb_run is not None:
            wandb_run.finish()



def vanishing_grad_analysis(args, config, x_train, y_train, x_val, y_val, NeuralNetwork=None):
    activation_functions = ["sigmoid", "tanh", "relu"]
    for act in activation_functions:
        print(f"\n{'='*50}")
        print(f"  Vanishing Gradient Analysis: {act.upper()}")
        print(f"{'='*50}")

        analysis_args = argparse.Namespace(
            dataset       = args.dataset,
            epochs        = args.epochs,
            batch_size    = args.batch_size,
            learning_rate = args.learning_rate,
            optimizer     = "adam",              # fixed: SGD to isolate activation effect
            loss          = args.loss,
            weight_decay  = args.weight_decay,
            num_neurons   = args.num_neurons,   # fixed: 3 layers × 128
            activation    = act,               # variable: activation function
            weight_init   = args.weight_init,
            seed          = args.seed,
            save_dir      = args.save_dir,
            model_save_path = f"vanishing_{act}.npy",
        )

        wandb_run = None
        if not args.no_wandb:
            wandb_run = wandb.init(
                project = args.wandb_project,
                entity  = args.wandb_entity,
                config  = {
                    "experiment"   : "2.4_vanishing_gradient_analysis",
                    "activation"   : act,
                    "epochs"       : analysis_args.epochs,
                    "batch_size"   : analysis_args.batch_size,
                    "learning_rate": analysis_args.learning_rate,
                    "num_neurons"  : analysis_args.num_neurons,
                    "dataset"      : analysis_args.dataset,
                },
                name   = f"vanishing_{act}",
                group  = "2.4_vanishing_gradient_analysis",   # groups all runs in W&B
                reinit = True,
            )

        model = NeuralNetwork(config, analysis_args)
        model.train(
            x_train, y_train,
            x_val,   y_val,
            epochs     = analysis_args.epochs,
            batch_size = analysis_args.batch_size,
            save_dir   = analysis_args.save_dir,
            wandb_run  = wandb_run,
        )

        if wandb_run is not None:
            wandb_run.finish()