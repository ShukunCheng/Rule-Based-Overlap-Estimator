import json
import warnings
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.metrics import confusion_matrix
from rule_based import RuleBased


def IoU(y_ture, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_ture, y_pred, labels=[0, 1]).ravel()
    iou = tp / (tp + fp + fn)
    acc = (tp + tn) / (tp + tn + fp + fn)
    return iou, acc


def thread_func(mean1, mean2, var, size, dim, eps, s, r, iou_array):
    warnings.filterwarnings('ignore')
    x1 = np.random.normal(mean1, var, (size, dim))
    x2 = np.random.normal(mean2, var, (size, dim))
    X = np.concatenate([x1, x2]).reshape(-1, dim)
    f1 = stats.multivariate_normal.pdf(x=X, mean=[mean1] * dim, cov=var)
    f2 = stats.multivariate_normal.pdf(x=X, mean=[mean2] * dim, cov=var)
    true_ov = (f1 > eps) & (f2 > eps)

    estimator = RuleBased(alpha=s, beta=r)
    y = np.concatenate([np.ones(len(x1)), np.zeros(len(x2))])
    try:
        estimator.fit(X, y)
        estimated_ov = estimator.predict(X)
        iou, acc = IoU(true_ov, estimated_ov)
        iou_array.append(iou)
    except:
        iou_array.append(0)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    eps = 0.05
    mean_array = [2.5, 1, 0.65, 0.5, 0.43, 0.39]
    var_array = [1, 1, 0.5, 0.33, 0.25, 0.2]
    mean1 = 0
    res = {}
    x_axis = []
    sample_size = 10
    step = 10
    max_size = 15000
    index = 6
    # Store x axis
    while sample_size < max_size:
        x_axis.append(sample_size)
        sample_size += step
        step *= 2

    for i in range(index):
        dim = i + 1
        a = 0.3
        b = 0.9
        print("Dimension: %s" % dim)
        var = var_array[i]
        mean2 = mean_array[i]
        res[dim] = {"mean": [], "var": []}
        sample_size = 10
        step = 10
        # Test for each sample size
        while sample_size < max_size:
            iou_array = []
            for _ in range(10):
                x1 = np.random.normal(mean1, var, (sample_size, dim))
                x2 = np.random.normal(mean2, var, (sample_size, dim))
                X = np.concatenate([x1, x2]).reshape(-1, dim)
                f1 = stats.multivariate_normal.pdf(x=X, mean=[mean1] * dim, cov=var)
                f2 = stats.multivariate_normal.pdf(x=X, mean=[mean2] * dim, cov=var)
                true_ov = (f1 > eps) & (f2 > eps)

                estimator = RuleBased(alpha=a, beta=b)
                y = np.concatenate([np.ones(len(x1)), np.zeros(len(x2))])
                try:
                    estimator.fit(X, y)
                    estimated_ov = estimator.predict(X)
                    iou, acc = IoU(true_ov, estimated_ov)
                    iou_array.append(iou)
                except:
                    iou_array.append(0)

            res[dim]["mean"].append(np.average(iou_array))
            res[dim]["var"].append(np.var(iou_array))
            sample_size += step
            step *= 2
    # Store data into json file
    with open("results/number_of_samples_with_best_hyper.json", "w") as file:
        json.dump(res, file, indent=4)

    markers = ['o', "^", '+', 'x', '*', '.', 'X']

    # Read from json file
    f = open("results/number_of_samples_with_best_hyper.json")
    data = json.load(f)
    plt.figure()
    for i in range(index):
        dim = i + 1
        mean = data[str(dim)]["mean"]
        var = data[str(dim)]["var"]
        if dim == 1:
            plt.plot(x_axis, mean, label="%s feature" % dim, marker=markers[i])
        else:
            plt.plot(x_axis, mean, label="%s features" % dim, marker=markers[i])

        plt.fill_between(x_axis, [mean[i] - var[i] for i in range(len(mean))],
                         [mean[i] + var[i] for i in range(len(mean))], alpha=0.5)
    plt.xlabel("Sample size")
    plt.ylabel("IoU")
    plt.xscale('symlog')
    plt.legend()
    plt.show()
