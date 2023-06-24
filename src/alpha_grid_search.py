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


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    eps = 0.05
    mean_array = [2.5, 1, 0.65, 0.5, 0.43, 0.39]
    var_array = [1, 1, 0.5, 0.33, 0.25, 0.2]
    mean1 = 0
    sample_sizes = [200, 200, 200, 1000, 1200, 10000]
    alpha_array = [round(x, 2) for x in np.arange(0.01, 1, 0.02)]
    res = {}
    index = 1

    for i in range(index):
        dim = i + 1
        print("Dimension: %s" % dim)
        mean2 = mean_array[i]
        var = var_array[i]
        sample_size = sample_sizes[i]
        res[dim] = {"br_mean": np.zeros(len(alpha_array)),
                    "br_var": np.zeros(len(alpha_array)),
                    "svm_mean": np.zeros(len(alpha_array)),
                    "svm_var": np.zeros(len(alpha_array))}
        for j in range(len(alpha_array)):
            a = alpha_array[j]
            print(a)
            br_array = []
            svm_array = []
            # Run 10 times and take average
            for _ in range(10):
                x1 = np.random.normal(mean1, var, (sample_size, dim))
                x2 = np.random.normal(mean2, var, (sample_size, dim))
                X = np.concatenate([x1, x2]).reshape(-1, dim)
                f1 = stats.multivariate_normal.pdf(x=X, mean=[mean1] * dim, cov=var)
                f2 = stats.multivariate_normal.pdf(x=X, mean=[mean2] * dim, cov=var)
                true_ov = (f1 > eps) & (f2 > eps)

                estimator = RuleBased(alpha=a)
                y = np.concatenate([np.ones(len(x1)), np.zeros(len(x2))])
                try:
                    estimator.fit(X, y)
                    estimated_ov = estimator.predict(X)
                    svm_ov = estimator.predict(X, use_density=True)
                    iou, _ = IoU(true_ov, estimated_ov)
                    svm_iou, _ = IoU(true_ov, svm_ov)
                    br_array.append(iou)
                    svm_array.append(svm_iou)
                except:
                    continue
                if len(br_array) != 0:
                    res[dim]["br_mean"][j] = np.average(br_array)
                    res[dim]["br_var"][j] = np.var(br_array)
                if len(svm_array) != 0:
                    res[dim]["svm_mean"][j] = np.average(svm_array)
                    res[dim]["svm_var"][j] = np.var(svm_array)
    # Store the res into file
    # np.save("../results/alpha_grid_search.npy", res)

    data = res
    # data = np.load("../results/alpha_grid_search.npy", allow_pickle=True).item()
    markers = ['o', '*', '+', 'x', '.', 'X']
    # Plot each dimension
    for i in range(index):
        dim = i + 1
        br_mean = data[dim]["br_mean"]
        br_var = data[dim]["br_var"]
        svm_mean = data[dim]["svm_mean"]
        svm_var = data[dim]["svm_var"]
        plt.figure()
        plt.plot(alpha_array, br_mean, label="Overrule", marker=markers[0])
        plt.plot(alpha_array, svm_mean, label="One Class SVM", marker=markers[1])
        plt.fill_between(alpha_array, [br_mean[i] - br_var[i] for i in range(len(br_mean))],
                         [br_mean[i] + br_var[i] for i in range(len(br_mean))], alpha=0.5)
        plt.fill_between(alpha_array, [svm_mean[i] - svm_var[i] for i in range(len(svm_mean))],
                         [svm_mean[i] + svm_var[i] for i in range(len(svm_mean))], alpha=0.5)
        plt.xlabel("alpha")
        plt.ylabel("IoU")
        plt.legend()
        plt.show()
        plt.clf()
        plt.cla()
        plt.close()
