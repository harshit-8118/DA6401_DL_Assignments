import argparse
import json
import os
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

    model = NeuralNetwork.load(
        weights_path      = model_path,
        config_path       = config_path,
        hyperparams_config = CONFIG          # ← the missing argument
    )

    _, (X_test, y_test) = load_dataset(args.dataset)
    # y_test is already one-hot from load_dataset

    results = evaluate_batched(model, X_test, y_test, batch_size=args.batch_size)

    print("\n── Test Set Metrics ─────────────────────")
    print(f"  Accuracy  : {results['accuracy']:.4f}")
    print(f"  Precision : {results['precision']:.4f}")
    print(f"  Recall    : {results['recall']:.4f}")
    print(f"  F1-Score  : {results['f1']:.4f}")
    print("\nConfusion Matrix:")
    print(results['confusion_matrix'])

    save_results(results, args.save_dir)
    return results


if __name__ == '__main__':
    main()