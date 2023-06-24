import numpy as np
from sklearn.metrics import confusion_matrix
from abstract import AbstractOverlapEstimator
from overrule.support import SVMSupportEstimator
from overrule.overrule import OverRule


class RuleBased(AbstractOverlapEstimator):
    def __init__(self,
                 overlap_estimator='support',
                 support_estimator=SVMSupportEstimator,
                 support_estimator_1=None,
                 alpha=0.1, beta=0.9,
                 n_ref_multiplier=1.,
                 support_kwargs={}, ruleset_kwargs={},
                 ruleset_estimator='bcs'):
        self.overlap_estimator = overlap_estimator
        self.support_estimator = support_estimator
        self.support_estimator_1 = support_estimator_1
        self.alpha = alpha
        self.beta = beta
        self.n_ref_multiplier = n_ref_multiplier
        self.support_kwargs = support_kwargs
        self.ruleset_kwargs = ruleset_kwargs
        self.ruleset_estimator = ruleset_estimator

        self.ov = OverRule(overlap_estimator, support_estimator, support_estimator_1, alpha, 1 - beta,
                           n_ref_multiplier, support_kwargs, ruleset_kwargs, ruleset_estimator)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the overlap estimator to the data.

        Args:
            X (np.ndarray): Array of shape (n_samples, n_features) containing the input data.
            y (np.ndarray): Array of shape (n_samples, ) containing the group labels for each data point.
        """
        self.ov.fit(X, y)

    def predict(self, X: np.ndarray, use_density=False) -> np.ndarray:
        """Predict whether each data point is in the overlap region.

        Args:
            X (np.ndarray): Array of shape (n_samples, n_features) containing the input data.

        Returns:
            np.ndarray: Array of shape (n_samples, ) with 0 or 1 for each data point.
                        0 means the data point is not in the overlap region,
                        and 1 means the data point is in the overlap region.
        """
        return self.ov.predict(X, use_density)

    def get_overlap_region(self) -> object:
        """Return the overlap region identified by the overlap estimator.
        #TODO Have to figure out what representation to use for the overlap region.

        Returns:
            object: The overlap region.
        """
        pass

    def score(self, X: np.ndarray, y: np.ndarray, score_type: str = "accuracy") -> float:
        """Compute the score of the model with X and y as test data.

        Args:
            X (np.ndarray): Array of shape (n_samples, n_features) containing the test data points.
            y (np.ndarray): Array of shape (n_samples, ) containing the group labels for each data point.
            score_type (str, optional): The type of score to compute. Defaults to "accuracy". # TODO Think about other scores.

        Returns:
            float: The score of the model.
        """
        prediction = self.predict(X)
        tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
        if score_type.lower() == "accuracy":
            return (tp + tn) / (tp + tn + fp + fn)
        elif score_type.lower() == "iou":
            return tp / (tp + fp + fn)
        else:
            pass

    def get_params(self, deep=False) -> dict:
        """Return the parameters of the overlap estimator. #TODO maybe not needed

        Returns:
            dict: The parameters of the overlap estimator.
        """
        return {"alpha_s": self.alpha, "alpha_r": self.beta}

    def set_params(self, **params) -> None:
        """Set the parameters of the overlap estimator. #TODO maybe not needed

        Args:
            **params: The parameters to set.
        """
        pass