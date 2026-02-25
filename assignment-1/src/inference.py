import argparse
import json
import wandb
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from ann import NeuralNetwork
from utils import load_dataset

CONFIG = {
    'beta': 0.9,
    'epsilon': 1e-8,
    'adam_beta1': 0.9,
    'adam_beta2': 0.999,
    'nadam_beta1': 0.9,
    'nadam_beta2': 0.999,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',  type=str, default='best_model.npy')
    parser.add_argument('--config_path', type=str, default='best_config.json')
    parser.add_argument('--dataset',     type=str, default='mnist', choices=['mnist', 'fashion_mnist'])
    parser.add_argument('--save_dir',    type=str, default='assignment-1/models')
    parser.add_argument('--batch_size',  type=int, default=512)
    # wandb arguments
    parser.add_argument('--wandb_project', type=str, default='DA6401_Assignment1')
    parser.add_argument('--wandb_entity',  type=str, default=None)
    parser.add_argument('--wandb_run_name',type=str, default='2.8_Error_analysis')
    parser.add_argument('--no_wandb',      action='store_true', help='Disable wandb logging')
    return parser.parse_args()


def evaluate_batched(model, X, y_onehot, batch_size=512):
    n = X.shape[0]
    all_probs = []

    for start in range(0, n, batch_size):
        X_b = X[start : start + batch_size]
        all_probs.append(model.predict_proba(X_b))

    probs      = np.vstack(all_probs)
    y_pred_lbl = np.argmax(probs,    axis=1)
    y_true_lbl = np.argmax(y_onehot, axis=1)

    acc  = accuracy_score(y_true_lbl, y_pred_lbl)
    prec = precision_score(y_true_lbl, y_pred_lbl, average='macro', zero_division=0)
    rec  = recall_score(y_true_lbl, y_pred_lbl, average='macro', zero_division=0)
    f1   = f1_score(y_true_lbl, y_pred_lbl, average='macro', zero_division=0)
    cm   = confusion_matrix(y_true_lbl, y_pred_lbl)

    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'confusion_matrix': cm}

def plot_confusion_matrix(cm, dataset_name):
    """Create and return a matplotlib figure of the confusion matrix."""
    n_classes = cm.shape[0]
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    fig.colorbar(im, ax=ax)

    # Annotate each cell
    thresh = cm.max() / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black',
                    fontsize=6)

    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(range(n_classes))
    ax.set_yticklabels(range(n_classes))
    ax.set_title(f'Confusion Matrix — {dataset_name} (Test Set)', fontsize=14, pad=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    plt.tight_layout()
    return fig

def save_results(results, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    out = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in results.items()}
    with open(os.path.join(save_dir, 'inference_results.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Results saved to '{save_dir}/inference_results.json'")


def main():
    args = parse_args()

    config_path = os.path.join(args.save_dir, args.config_path)
    model_path  = os.path.join(args.save_dir, args.model_path)

    # ── Init wandb ────────────────────────────────────────────────────────────
    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project = args.wandb_project,
            entity  = args.wandb_entity,
            name    = args.wandb_run_name,
            config  = {
                'dataset':    args.dataset,
                'batch_size': args.batch_size,
                'model_path': args.model_path,
            },
        )

    model = NeuralNetwork.load(
        weights_path      = model_path,
        config_path       = config_path,
        hyperparams_config = CONFIG          # ← the missing argument
    )

    (X_train, y_train), (X_test, y_test) = load_dataset(args.dataset)
    # y_test is already one-hot from load_dataset

    results_train = evaluate_batched(model, X_train, y_train, batch_size=args.batch_size)

    print("\n── Train Set Metrics ─────────────────────")
    print(f"  Accuracy  : {results_train['accuracy']:.4f}")
    print(f"  Precision : {results_train['precision']:.4f}")
    print(f"  Recall    : {results_train['recall']:.4f}")
    print(f"  F1-Score  : {results_train['f1']:.4f}")
    print("\nConfusion Matrix:")
    print(results_train['confusion_matrix'])

    results = evaluate_batched(model, X_test, y_test, batch_size=args.batch_size)
    print("\n── Test Set Metrics ──────────────────────")
    print(f"  Accuracy  : {results['accuracy']:.4f}")
    print(f"  Precision : {results['precision']:.4f}")
    print(f"  Recall    : {results['recall']:.4f}")
    print(f"  F1-Score  : {results['f1']:.4f}")
    print("\nConfusion Matrix:")
    print(results['confusion_matrix'])

    if use_wandb:
        # Confusion matrix figure (test set only)
        fig = plot_confusion_matrix(results['confusion_matrix'], args.dataset)
        wandb.log({'test/confusion_matrix': wandb.Image(fig)})
        plt.close(fig)

        wandb.finish()

    save_results(results, args.save_dir)
    return results


if __name__ == '__main__':
    main()