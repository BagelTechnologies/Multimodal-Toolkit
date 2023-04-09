import math
import numpy as np
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    roc_auc_score,
    f1_score,
    confusion_matrix,
    matthews_corrcoef,
    mean_squared_error,
    mean_absolute_error,
)
from scipy.special import softmax


def calc_classification_metrics(predictions, labels):
    pred_labels = np.argmax(predictions, axis=1)
    if len(np.unique(labels)) == 2:  # binary classification
        pred_scores = softmax(predictions, axis=1)[:, 1]
        roc_auc_pred_score = roc_auc_score(labels, pred_scores)
        precisions, recalls, thresholds = precision_recall_curve(labels, pred_scores)
        fscore = (2 * precisions * recalls) / (precisions + recalls)
        fscore[np.isnan(fscore)] = 0
        ix = np.argmax(fscore)
        threshold = thresholds[ix].item()
        pr_auc = auc(recalls, precisions)
        tn, fp, fn, tp = confusion_matrix(labels, pred_labels, labels=[0, 1]).ravel()
        result = {
            "roc_auc": roc_auc_pred_score,
            "threshold": threshold,
            "pr_auc": pr_auc,
            "recall": recalls[ix].item(),
            "precision": precisions[ix].item(),
            "f1": fscore[ix].item(),
            "tn": tn.item(),
            "fp": fp.item(),
            "fn": fn.item(),
            "tp": tp.item(),
        }
    else:
        topn_acc = {}
        for top_n in [1, 2, 3, 5]:
            adjusted_predictions = []
            for i in range(predictions.shape[0]):
                probs_i = predictions[i, :]
                top_n_preds_i = np.argsort(probs_i)[-top_n:][::-1]
                adjusted_predictions.append(labels[i] if labels[i] in top_n_preds_i else top_n_preds_i[0])
            topn_acc[f'acc-top-{top_n}'] = (np.array(adjusted_predictions) == labels).mean()

        acc = (pred_labels == labels).mean()
        f1_micro = f1_score(y_true=labels, y_pred=pred_labels, average="micro")
        f1_macro = f1_score(y_true=labels, y_pred=pred_labels, average="macro")
        f1_weighted = f1_score(y_true=labels, y_pred=pred_labels, average="weighted")

        result = {
            "acc": acc,
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "mcc": matthews_corrcoef(labels, pred_labels),
        }
        result = {**result, **topn_acc}

    return result


def calc_regression_metrics(preds, labels):
    mse = mean_squared_error(labels, preds)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(labels, preds)
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
    }
