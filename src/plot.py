import warnings
import numpy as np
from matplotlib import pyplot as plt, colors
from scipy import stats


if __name__ == '__main__':
    cmap_matrix = [[1, 1, 1], [1, 1, 0]]
    cm = colors.ListedColormap(cmap_matrix)
    warnings.filterwarnings('ignore')
    eps = 0.05
    mean1 = 0.5
    mean2 = -0.5
    out_mean1 = -1.5
    out_var1 = 1
    out_size1 = 20
    var = 1
    sample_size = 200
    dim = 2
    bin_size = 200
    lower, upper, step = -4, 4, 1 / bin_size
    # Get overlap region
    xs, ys = np.mgrid[lower:upper:step, lower:upper:step]
    ts = np.dstack((xs, ys))
    f1_distribution = stats.multivariate_normal.pdf(x=ts, mean=[mean1] * dim, cov=var, allow_singular=True)
    f2_distribution = stats.multivariate_normal.pdf(x=ts, mean=[mean2] * dim, cov=var, allow_singular=True)
    outlier_distribution1 = stats.multivariate_normal.pdf(x=ts, mean=[out_mean1] * dim, cov=out_var1, allow_singular=True)
    class_ov = (f1_distribution > eps) & (f2_distribution > eps)
    outlier1 = np.random.uniform(-4, 4, (out_size1, dim))

    plt.figure()
    # Overlap data from boolean rules
    plt.scatter(outlier1[:, 0], outlier1[:, 1], alpha=0.5, marker="^", color="red")
    plt.contour(xs, ys, f1_distribution, cmap='Reds', label="x1")
    plt.contour(xs, ys, f2_distribution, cmap='Blues', label="x2")
    plt.imshow(class_ov.T, extent=[lower, upper, lower, upper], origin="lower", cmap=cm)
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()