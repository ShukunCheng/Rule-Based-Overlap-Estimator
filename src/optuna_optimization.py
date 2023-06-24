import warnings
import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix
import optuna
import json
from rule_based import RuleBased


def IoU(y_ture, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_ture, y_pred, labels=[0, 1]).ravel()
    iou = tp / (tp + fp + fn)
    acc = (tp + tn) / (tp + tn + fp + fn)
    return iou, acc


def objective(trial, m1, m2, v1, s, d):
    warnings.filterwarnings('ignore')
    alpha = trial.suggest_float("alpha", 0.01, 1)
    beta = trial.suggest_float("beta", 0.01, 1)
    iou_array = []
    for _ in range(5):
        x1 = np.random.normal(m1, v1, (s, d))
        x2 = np.random.normal(m2, v1, (s, d))
        y = np.concatenate([np.ones(len(x1)), np.zeros(len(x2))])

        X = np.concatenate([x1, x2]).reshape(-1, d)
        f1 = stats.multivariate_normal.pdf(x=X, mean=[m1] * d, cov=v1)
        f2 = stats.multivariate_normal.pdf(x=X, mean=[m2] * d, cov=v1)
        true_ov = (f1 > eps) & (f2 > eps)

        estimator = RuleBased(alpha=alpha, beta=beta)
        try:
            estimator.fit(X, y)
            estimated_ov = estimator.predict(X)
            iou, acc = IoU(true_ov, estimated_ov)
            iou_array.append(iou)
        except:
            continue
    if len(iou_array) == 0:
        return -1
    else:
        return np.average(iou_array)


if __name__ == '__main__':
    eps = 0.05
    mean_array = [2.5, 1, 0.65, 0.5, 0.43, 0.39]
    var_array = [1, 1, 0.5, 0.33, 0.25, 0.2]
    mean1 = 0
    sample_sizes = [200, 200, 200, 1000, 1200, 15000]
    parameters = {}
    index = np.concatenate([[0.01], np.arange(0.05, 1.05, 0.05)])
    search_space = {
        'alpha': [round(x, 2) for x in index],
        'beta': [round(x, 2) for x in index]
    }

    for i in range(6):
        mean2 = mean_array[i]
        var = var_array[i]
        dim = i + 1
        samples = sample_sizes[i]

        # Grid search using optuna
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.GridSampler(search_space))
        study.optimize(lambda trial: objective(trial, mean1, mean2, var, samples, dim))
        fig = optuna.visualization.plot_contour(study, target_name="IoU")
        fig.update_layout(width=800, height=800)
        fig.show()
        parameters[dim] = study.best_trial.params

    with open("../results/parameters.json", "w") as f:
        json.dump(parameters, f)
