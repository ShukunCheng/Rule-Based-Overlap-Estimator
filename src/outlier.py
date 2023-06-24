import warnings
import numpy as np
from matplotlib import pyplot as plt, colors
from scipy import stats
from sklearn.metrics import confusion_matrix
from rule_based import RuleBased


def IoU(y_ture, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_ture, y_pred, labels=[0, 1]).ravel()
    iou = tp / (tp + fp + fn)
    acc = (tp + tn) / (tp + tn + fp + fn)
    return iou, acc


if __name__ == '__main__':
    cmap_matrix = [[1, 1, 1], [1, 1, 0]]
    cm = colors.ListedColormap(cmap_matrix)
    warnings.filterwarnings('ignore')
    eps = 0.05
    mean_array = [2.5, 1, 0.65, 0.5, 0.43, 0.39]
    var_array = [1, 1, 0.5, 0.33, 0.25, 0.2]
    # 1d outlier mean
    out_mean1 = 1
    # 2d outlier mean
    # out_mean1 = -1.5
    out_var1 = 1
    var = 1
    sample_size = 200
    outlier_array = np.arange(1, 10)
    # Test the dimension you want
    dim = 2
    mean1 = mean_array[dim - 1] / 2
    mean2 = -mean1
    bin_size = 50
    lower, upper, step = -4, 4, 1 / bin_size
    alpha_s = 0.3
    alpha_r = 0.5
    res = {"br_mean": np.zeros(len(outlier_array)), "br_var": np.zeros(len(outlier_array)),
           "svm_mean": np.zeros(len(outlier_array)), "svm_var": np.zeros(len(outlier_array))}

    for i in range(len(outlier_array)):
        size = int(sample_size * outlier_array[i] / 100)
        print("Outlier percent: %s, Amount: %s" % (outlier_array[i], size))
        br_array = []
        svm_array = []
        for _ in range(10):
            x1 = np.random.normal(mean1, var, (sample_size, dim))
            x2 = np.random.normal(mean2, var, (sample_size, dim))
            X = np.concatenate([x1, x2]).reshape(-1, dim)
            f1 = stats.multivariate_normal.pdf(x=X, mean=[mean1] * dim, cov=var)
            f2 = stats.multivariate_normal.pdf(x=X, mean=[mean2] * dim, cov=var)
            true_ov = (f1 > eps) & (f2 > eps)
            # Generate outlier data
            outlier1 = np.random.uniform(lower, upper, (size, dim))

            try:
                estimator = RuleBased(alpha=0.1)
                X = np.concatenate([X, outlier1.reshape(-1, dim)])
                y = np.concatenate([np.ones(len(x1)), np.zeros(len(x2)), np.ones(len(outlier1))])
                estimator.fit(X, y)

                svm_ov = estimator.predict(X, True)
                svm_ov_data = X[svm_ov == 1]
                br_ov = estimator.predict(X)
                br_ov_data = X[br_ov == 1]
                true_ov = np.concatenate([true_ov, np.zeros(len(outlier1), dtype=bool)])
                svm_iou, svm_acc = IoU(true_ov, svm_ov)
                br_iou, br_acc = IoU(true_ov, br_ov)
                br_array.append(br_iou)
                svm_array.append(svm_iou)
            except:
                continue
        res["br_mean"][i] = np.average(br_array)
        res["br_var"][i] = np.var(br_array)
        res["svm_mean"][i] = np.average(svm_array)
        res["svm_var"][i] = np.var(svm_array)
    np.save("../results/outliers.npy", res)

    markers = ['o', '+', 'x', '*', '.', 'X']

    data = np.load("../results/outliers.npy", allow_pickle=True).item()
    plt.figure()
    br_mean = data["br_mean"]
    br_var = data["br_var"]
    svm_mean = data["svm_mean"]
    svm_var = data["svm_var"]
    plt.plot(outlier_array, br_mean, label="Overrule", marker=markers[0])
    plt.plot(outlier_array, svm_mean, label="One Class SVM", marker=markers[1])
    plt.fill_between(outlier_array, [br_mean[i] - br_var[i] for i in range(len(br_mean))],
                     [br_mean[i] + br_var[i] for i in range(len(br_mean))], alpha=0.5)
    plt.fill_between(outlier_array, [svm_mean[i] - svm_var[i] for i in range(len(svm_mean))],
                     [svm_mean[i] + svm_var[i] for i in range(len(svm_mean))], alpha=0.5)
    plt.xlabel("Percent of outlier samples")
    plt.ylabel("IoU")
    plt.legend()
    plt.show()



