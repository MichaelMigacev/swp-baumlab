import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    balanced_accuracy_score, precision_score, recall_score,
    cohen_kappa_score, confusion_matrix
)
from model import Encoder, load_data


def binarize(labels, threshold=30):
    return (np.array(labels) >= threshold).astype(int)


def compute_metrics(y_true, y_pred, threshold=30):
    y_true_bin = binarize(y_true, threshold)
    y_pred_bin = binarize(y_pred, threshold)
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin).ravel()

    metrics = {
        "ROC AUC": roc_auc_score(y_true_bin, y_pred),
        "PR AUC": average_precision_score(y_true_bin, y_pred),
        "ACC": accuracy_score(y_true_bin, y_pred_bin),
        "BACC": balanced_accuracy_score(y_true_bin, y_pred_bin),
        "PREC": precision_score(y_true_bin, y_pred_bin),
        "TPR (Sensitivity)": recall_score(y_true_bin, y_pred_bin),
        "TNR (Specificity)": tn / (tn + fp),
        "Kappa": cohen_kappa_score(y_true_bin, y_pred_bin),
    }
    return metrics


def find_best_threshold(y_true, y_pred):
    thresholds = np.linspace(0, 100, 500)
    best_bacc = -1
    best_thresh = 30
    for t in thresholds:
        bacc = balanced_accuracy_score(binarize(y_true), binarize(y_pred, t))
        if bacc > best_bacc:
            best_bacc = bacc
            best_thresh = t
    return best_thresh, best_bacc


def evaluate_model(model_path, data_file, layers, dropout, batch_size=256):
    # Load data
    _, _, _, X_test, _, _, _, y_test, _, _, _ = load_data(data_file)
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Load model
    input_size = X_test.shape[1]
    encoder = Encoder(
        input_size=input_size,
        layers=layers,
        input_dropout=dropout["input"],
        hidden_dropout=dropout["hidden"]
    )
    encoder.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    encoder.eval()

    # Generate predictions
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in test_loader:
            preds = encoder(x).squeeze().numpy()
            all_preds.extend(preds)
            all_targets.extend(y.numpy())

    # Optimize threshold
    best_thresh, best_bacc = find_best_threshold(all_targets, all_preds)
    print(f"Best threshold (BACC): {best_thresh:.2f}, BACC: {best_bacc:.3f}")

    # Compute metrics with best threshold
    metrics = compute_metrics(all_targets, all_preds, threshold=best_thresh)
    print("\nPerformance Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")


if __name__ == "__main__":
    model_path = "saved_models/test0val1normtanh_norm.p_model.pt"
    data_file = "test0val1normtanh_norm.p.gz"
    layers = [8192, 4096, 1]
    dropout = {"input": 0.2, "hidden": 0.5}

    evaluate_model(model_path, data_file, layers, dropout)
    
    
"""

| Metric                   | My Model   | DNN Model   | Comparison          | Interpretation                                                                 |
| ------------------------ | ---------- | ----------- | ------------------- | ------------------------------------------------------------------------------ |
| ROC AUC                  | 0.843      | 0.90 ± 0.03 | DNN performs better | Very good. Shows strong ability to distinguish between classes.                |
| PR AUC                   | 0.641      | 0.59 ± 0.06 | My model better     | Moderate. Indicates the model struggles a bit more with the positive class.    |
| Accuracy (ACC)           | 0.838      | 0.92 ± 0.03 | DNN performs better | High overall accuracy. However, this can be misleading in imbalanced datasets. |
| Balanced Accuracy (BACC) | 0.740      | 0.76 ± 0.03 | DNN slightly better | Fair. Reflects imbalance in TPR and TNR.                                       |
| Precision (PREC)         | 0.653      | 0.56 ± 0.11 | My model better     | Moderate. Of the predicted positives, 65.3% were correct.                      |
| Sensitivity (TPR)        | 0.564      | 0.57 ± 0.09 | Roughly equal       | Weak. The model misses a significant portion of true positives.                |
| Specificity (TNR)        | 0.916      | 0.95 ± 0.03 | DNN performs better | Excellent. The model is very good at identifying true negatives.               |
| Cohen’s Kappa            | 0.504      | 0.51 ± 0.04 | Roughly equal       | Moderate agreement. Somewhat better than random classification.                |


"""
