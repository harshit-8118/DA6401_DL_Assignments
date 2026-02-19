"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
from .neural_layer import NeuralLayer
from .objective_functions import OBJECTIVE
from .activations import ACTIVATIONS
from .optimizers import OPTIMIZERS, NAG
import numpy as np
import json
import os
import argparse


def compute_metrics(y_true_labels, y_pred_labels, num_classes=10):
    accuracy = float(np.mean(y_true_labels == y_pred_labels))

    precision_per_class = np.zeros(num_classes)
    recall_per_class    = np.zeros(num_classes)

    for c in range(num_classes):
        tp = np.sum((y_pred_labels == c) & (y_true_labels == c))
        fp = np.sum((y_pred_labels == c) & (y_true_labels != c))
        fn = np.sum((y_pred_labels != c) & (y_true_labels == c))
        precision_per_class[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall_per_class[c]    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    precision = float(np.mean(precision_per_class))
    recall    = float(np.mean(recall_per_class))
    denom     = precision + recall
    f1        = float(2 * precision * recall / denom) if denom > 0 else 0.0

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


class NeuralNetwork:

    def __init__(self, config, cli_args):
        self.config       = config
        self.cli_args     = cli_args
        self.layers       = []
        self.weight_decay = cli_args.weight_decay

        self.loss, self.loss_grad = OBJECTIVE[cli_args.loss]
        self._best_val_f1 = -1.0
        self._best_epoch  = 0

        layer_sizes = [784] + cli_args.num_neurons
        for i in range(len(layer_sizes) - 1):
            self.layers.append(NeuralLayer(
                input_size  = layer_sizes[i],
                output_size = layer_sizes[i + 1],
                activation  = cli_args.activation,
                weight_init = cli_args.weight_init,
                layer_name  = 'hidden'
            ))

        self.layers.append(NeuralLayer(
            input_size  = layer_sizes[-1],
            output_size = 10,
            activation  = 'identity',
            weight_init = cli_args.weight_init,
            layer_name  = 'output'
        ))

        opt_name     = cli_args.optimizer
        opt_cls      = OPTIMIZERS[opt_name]
        self.is_nag  = (opt_name == 'nag')

        if opt_name == 'sgd':
            self.optimizer = opt_cls(learning_rate=cli_args.learning_rate)
        elif opt_name in ('momentum', 'nag'):
            self.optimizer = opt_cls(learning_rate=cli_args.learning_rate,
                                     beta=config['beta'])
        elif opt_name == 'rmsprop':
            self.optimizer = opt_cls(learning_rate=cli_args.learning_rate,
                                     beta=config['beta'],
                                     epsilon=config['epsilon'])
        elif opt_name == 'adam':
            self.optimizer = opt_cls(learning_rate=cli_args.learning_rate,
                                     beta1=config['adam_beta1'],
                                     beta2=config['adam_beta2'],
                                     epsilon=config['epsilon'])
        elif opt_name == 'nadam':
            self.optimizer = opt_cls(learning_rate=cli_args.learning_rate,
                                     beta1=config['nadam_beta1'],
                                     beta2=config['nadam_beta2'],
                                     epsilon=config['epsilon'])

    #  Forward 

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out  # raw logits

    def predict_proba(self, X):
        softmax_fn, _ = ACTIVATIONS['softmax']
        return softmax_fn(self.forward(X))

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    #  Backward 

    def backward(self, y_true, y_pred):
        loss_val = self.loss(y_true=y_true, y_pred=y_pred)
        delta    = self.loss_grad(y_true=y_true, y_pred=y_pred)
        for layer in reversed(self.layers):
            delta = layer.backward(delta, weight_decay=self.weight_decay)
        return loss_val

    def update_weights(self):
        self.optimizer.update(self.layers)

    #  Evaluate 

    def evaluate(self, X, y_onehot, split_name='val'):
        probs      = self.predict_proba(X)
        loss       = self.loss(y_true=y_onehot, y_pred=probs)
        y_pred_lbl = np.argmax(probs,    axis=1)
        y_true_lbl = np.argmax(y_onehot, axis=1)
        metrics    = compute_metrics(y_true_lbl, y_pred_lbl)
        metrics['loss'] = float(loss)
        return metrics

    #  Train 

    def train(self, X_train, y_train, X_val, y_val,
              epochs, batch_size, save_dir='.', wandb_run=None):
        n = X_train.shape[0]

        print(f"\n{''*80}")
        print(f"  {n} samples | batch={batch_size} | epochs={epochs} | "
              f"opt={self.cli_args.optimizer} | lr={self.cli_args.learning_rate} | "
              f"wd={self.weight_decay}")
        print(f"{''*80}")

        for epoch in range(epochs):
            idx    = np.random.permutation(n)
            X_shuf = X_train[idx]
            y_shuf = y_train[idx]

            for start in range(0, n, batch_size):
                X_b = X_shuf[start : start + batch_size]
                y_b = y_shuf[start : start + batch_size]

                if self.is_nag:
                    self.optimizer.lookahead(self.layers)

                probs = self.predict_proba(X_b)
                loss  = self.backward(y_true=y_b, y_pred=probs)

                if self.is_nag:
                    self.optimizer.restore(self.layers)

                self.update_weights()

            sample_idx = np.random.choice(n, size=min(5000, n), replace=False)
            train_m = self.evaluate(X_train[sample_idx], y_train[sample_idx], 'train')
            val_m   = self.evaluate(X_val, y_val, 'val')

            print('-' * 50)
            print(f"epoch: [{epoch+1}/{epochs}]")
            print(f"Train ===== Loss: {train_m['loss']:.6f} | Acc: {train_m['accuracy']:.6f} | F1: {train_m['f1']:.6f}")
            print(f"Valid ===== Loss: {val_m['loss']:.6f}   | Acc: {val_m['accuracy']:.6f} | F1: {val_m['f1']:.6f}")

            if wandb_run is not None:
                wandb_run.log({
                    'epoch'           : epoch + 1,
                    'train/loss'      : train_m['loss'],
                    'train/accuracy'  : train_m['accuracy'],
                    'train/precision' : train_m['precision'],
                    'train/recall'    : train_m['recall'],
                    'train/f1'        : train_m['f1'],
                    'val/loss'        : val_m['loss'],
                    'val/accuracy'    : val_m['accuracy'],
                    'val/precision'   : val_m['precision'],
                    'val/recall'      : val_m['recall'],
                    'val/f1'          : val_m['f1'],
                    **{f'grad_norm/layer_{i}': float(np.linalg.norm(l.grad_W))
                       for i, l in enumerate(self.layers) if l.grad_W is not None},
                })

            if val_m['f1'] > self._best_val_f1:
                self._best_val_f1 = val_m['f1']
                self._best_epoch  = epoch + 1
                self.save_model(save_dir)

        print(f"\n  Best val F1={self._best_val_f1:.4f} at epoch {self._best_epoch}")
        print(f"{''*80}\n")

    #  Save 

    def save_model(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        params = [{'W': l.W.copy(), 'b': l.b.copy()} for l in self.layers]
        weights_filename = getattr(self.cli_args, 'model_save_path', 'best_model.npy')
        np.save(os.path.join(save_dir, weights_filename), params, allow_pickle=True)

        # Config: only plain Python types, only what inference.py needs
        best_config = {
            #  Architecture (must match to reconstruct layers) 
            'num_neurons'  : self.cli_args.num_neurons,   # list[int]
            'activation'   : self.cli_args.activation,    # str
            'weight_init'  : self.cli_args.weight_init,   # str
            #  Training settings (for optimizer reconstruction) 
            'optimizer'    : self.cli_args.optimizer,     # str
            'learning_rate': self.cli_args.learning_rate, # float
            'weight_decay' : float(self.weight_decay),    # float
            'loss'         : self.cli_args.loss,          # str
            #  Metadata 
            'best_val_f1'  : float(self._best_val_f1),
            'best_epoch'   : int(self._best_epoch),
            'dataset'      : getattr(self.cli_args, 'dataset', 'mnist'),
        }

        with open(os.path.join(save_dir, 'best_config.json'), 'w') as f:
            json.dump(best_config, f, indent=2)

        print(f"   → Saved to '{save_dir}/' "
              f"(val_f1={self._best_val_f1:.4f}, epoch={self._best_epoch})")

    #  Load 

    @classmethod
    def load(cls, weights_path, config_path, hyperparams_config):
        with open(config_path, 'r') as f:
            cfg = json.load(f)

        cli_args = argparse.Namespace(
            num_neurons   = cfg['num_neurons'],
            activation    = cfg['activation'],
            weight_init   = cfg['weight_init'],
            optimizer     = cfg['optimizer'],
            learning_rate = cfg['learning_rate'],
            weight_decay  = cfg['weight_decay'],
            loss          = cfg['loss'],
            model_save_path = os.path.basename(weights_path),
            dataset       = cfg.get('dataset', 'mnist'),
        )

        # Step 3: build the network with correct signature
        model = cls(hyperparams_config, cli_args)

        # Step 4: overwrite random weights with saved weights
        params = np.load(weights_path, allow_pickle=True)
        for layer, p in zip(model.layers, params):
            layer.W = p['W'].copy()
            layer.b = p['b'].copy()

        print(f"Model loaded from '{weights_path}'")
        print(f"  Architecture : 784 → {' → '.join(str(n) for n in cfg['num_neurons'])} → 10")
        print(f"  Best val F1  : {cfg.get('best_val_f1', 'N/A')}")

        return model

    #  Utilities 

    def layer_gradient_norms(self):
        return [float(np.linalg.norm(l.grad_W)) if l.grad_W is not None else 0.0
                for l in self.layers]