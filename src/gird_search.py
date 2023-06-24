import warnings
import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix
from rule_based import RuleBased


def IoU(y_ture, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_ture, y_pred, labels=[0, 1]).ravel()
    iou = tp / (tp + fp + fn)
    acc = (tp + tn) / (tp + tn + fp + fn)
    return iou, acc


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    # define some variables
    sample_sizes = [200, 200, 200, 1000, 1200]
    eps = 0.05
    mean_array = [2.5, 1, 0.65, 0.5, 0.43, 0.39]
    var_array = [1, 1, 0.5, 0.33, 0.25, 0.2]
    mean1 = 0
    index = np.concatenate([[0.01], np.arange(0.05, 1.05, 0.05)])
    alpha_array = beta_array = [round(x, 2) for x in index]
    res = {}

    res["alpha_s_array"] = alpha_array
    res["alpha_r_array"] = beta_array

    for dim in range(1, 6):
        print("Dimension: %s" % dim)
        res[dim] = {"iou": np.zeros((len(alpha_array), len(beta_array))),
                    "iou_var": np.zeros((len(alpha_array), len(beta_array))),
                    "acc": np.zeros((len(alpha_array), len(beta_array))),
                    "acc_var": np.zeros((len(alpha_array), len(beta_array)))}

        for i in range(len(alpha_array)):
            alpha = alpha_array[i]
            print("alpha_s: %s" % alpha)
            for j in range(len(beta_array)):
                beta = beta_array[j]
                print("alpha_r: %s" % beta)
                sample = sample_sizes[dim - 1]
                mean2 = mean_array[dim - 1]
                var = var_array[dim - 1]
                iou_array = []
                acc_array = []

                for _ in range(10):
                    # Find true overlap data
                    x1 = np.random.normal(mean1, var, (sample, dim))
                    x2 = np.random.normal(mean2, var, (sample, dim))
                    X = np.concatenate([x1, x2]).reshape(-1, dim)
                    f1 = stats.multivariate_normal.pdf(x=X, mean=[mean1] * dim, cov=var)
                    f2 = stats.multivariate_normal.pdf(x=X, mean=[mean2] * dim, cov=var)
                    true_ov = (f1 > eps) & (f2 > eps)

                    # Estimate overlap data
                    estimator = RuleBased(alpha=alpha, beta=beta)
                    y = np.concatenate([np.ones(len(x1)), np.zeros(len(x2))])
                    try:
                        estimator.fit(X, y)
                        estimated_ov = estimator.predict(X)
                        iou, acc = IoU(true_ov, estimated_ov)
                        iou_array.append(iou)
                        acc_array.append(acc)
                    except:
                        continue

                if len(iou_array) == 0:
                    res[dim]["iou"][i][j] = -1
                    res[dim]["iou_var"][i][j] = -1
                else:
                    res[dim]["iou"][i][j] = np.average(iou_array)
                    res[dim]["iou_var"][i][j] = np.var(iou_array)
                if len(acc_array) == 0:
                    res[dim]["acc"][i][j] = -1
                    res[dim]["acc_var"][i][j] = -1
                else:
                    res[dim]["acc"][i][j] = np.average(acc_array)
                    res[dim]["acc_var"][i][j] = np.var(acc_array)
    np.save("results/grid_search.npy", res)
