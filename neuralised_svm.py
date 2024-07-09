import numpy as np
from sklearn.svm import SVC
from scipy.special import logsumexp as logsumexp
from scipy.special import softmax as softmax
from sklearn.metrics.pairwise import rbf_kernel as rbf_kernel
import copy

class neuralised_svm(SVC):

    def __init__(self, svc):
        """
        Init for the neuralised SVM. Implementation assumes as input a trained sklearn.svm SVC object.
        :param svc:
        """
        self.original_svc = svc
        self.intercept_ = svc.intercept_
        self.gamma = svc.gamma

        self.alphas_pos = svc.dual_coef_[0][svc.dual_coef_[0] > 0]
        self.alphas_neg = np.abs(svc.dual_coef_[0][svc.dual_coef_[0] < 0])

        self.x_sup_pos = svc.support_vectors_[svc.dual_coef_[0] > 0]
        self.x_sup_neg = svc.support_vectors_[svc.dual_coef_[0] < 0]

        self.num_pos = len(self.alphas_pos)
        self.num_neg = len(self.alphas_neg)


    def compute_z(self, x: np.array, with_intercept: bool=False) -> np.array:
        """
        Compute the support vector distance terms to the input x.
        :param x:
        :param with_intercept:
        :return:
        """
        if with_intercept:
            raise NotImplementedError("Intercept not implemented yet")
        sv_pos_diff = x[:, None] - self.x_sup_pos[None]
        sv_pos_sq_distance_norm = np.linalg.norm(sv_pos_diff, axis=2) ** 2
        z_pos = sv_pos_sq_distance_norm - np.log(self.alphas_pos)[None] / self.gamma

        sv_neg_diff = x[:, None] - self.x_sup_neg[None]
        sv_neg_sq_distance_norm = np.linalg.norm(sv_neg_diff, axis=2) ** 2
        z_neg = sv_neg_sq_distance_norm - np.log(self.alphas_neg)[None] / self.gamma

        return z_pos, z_neg


    def forward(self, x: np.array, with_intercept: bool=False) -> np.array:
        """
        Compute the forward pass of the neuralised SVM. Not to be confused with the decision function of the original
        SVM, as the intercept is not yet incorporated. Therefore, the sign of the neuralised output is not
        necessarily the same as the sign of the decision function of the original SVM.
        :param x:
        :param with_intercept: flag for later implementation of intercept
        :return:
        """

        if with_intercept:
            raise NotImplementedError("Intercept not implemented yet")

        z_pos, z_neg = self.compute_z(x, with_intercept=with_intercept)

        g = logsumexp(-self.gamma * z_pos, axis=1) - logsumexp(-self.gamma * z_neg, axis=1)
        return g


    def explain(
            self, x, first_rule: str="GI", eta: float=.1,
            with_intercept: bool=False, predicted_class=None,
            reweight_explanation=False) -> np.array:
        """
        The neuralised LRP implementation. Per default, the first rule is set to "GI" (Gradient * Input).
        Alternatively, the first rule can be set to "midpoint" which sets as reference point the midpoint between the
        support vectors of the detection neurons. Thus, local information can be incorporated, stabalising the explanation.
        The third option is "hybrid" which is a combination of the two. The hybrid rule is modified via the eta parameter.
        Eta = 0.1 tends to work well.

        Please note that the rules performance depend heavily on the scaling of the input data. If the input data is
        N(0, 1) standardised GI tends to perform well. If the input data is not standardised, the midpoint rule tends
        to perform better.

        As for the signs of the explanation: Positive values indicate that the feature contributes to the positive class,
        negative values indicate that the feature contributes to the negative class. It is not necessary to incorporate
        the class sign of the original SVM, as the explanation is already class specific.

        # param x which is a numpy array
        :param np.array x: The input data for which the explanation is to be computed. Either give a single data point
        or the entire explained dataset.
        :param str first_rule: The first rule to be applied in the LRP explanation. Options are "GI", "midpoint" and "hybrid".
        :param float eta: The weight of the hybrid rule. Only relevant if first_rule is set to "hybrid".
        :param bool with_intercept:
        :return np.array R: The explanation for the input data x.
        """
        if with_intercept:
            raise NotImplementedError("Intercept not implemented yet")
        z_pos, z_neg = self.compute_z(x, with_intercept=with_intercept)

        p_pos = softmax(-self.gamma * z_pos, axis=1)
        p_neg = softmax(-self.gamma * z_neg, axis=1)

        if first_rule == "GI":
            weighted_mean_sv_pos = np.einsum("ni, id -> nd", p_pos, self.x_sup_pos)
            weighted_mean_sv_neg = np.einsum("nj, jd -> nd", p_neg, self.x_sup_neg)

            R = x * 2 * (weighted_mean_sv_pos - weighted_mean_sv_neg)

        elif first_rule == "midpoint":
            squared_diff_pos = (x[:, None] - self.x_sup_pos[None]) ** 2
            squared_diff_neg = (x[:, None] - self.x_sup_neg[None]) ** 2

            weighted_mean_squared_diff_pos = np.einsum("ni, nid -> nd", p_pos, squared_diff_pos)
            weighted_mean_squared_diff_neg = np.einsum("nj, njd -> nd", p_neg, squared_diff_neg)

            R = weighted_mean_squared_diff_neg - weighted_mean_squared_diff_pos

        elif first_rule == "mean_midpoint":
            weighted_mean_sv_pos = np.einsum("ni, id -> nd", p_pos, self.x_sup_pos)
            weighted_mean_sv_neg = np.einsum("nj, jd -> nd", p_neg, self.x_sup_neg)

            mean_midpoint = (weighted_mean_sv_pos + weighted_mean_sv_neg) / 2
            R = 2 * (weighted_mean_sv_pos - weighted_mean_sv_neg) * (x - mean_midpoint)

        elif first_rule == "hybrid":
            R_GI = self.explain(x, first_rule="GI", with_intercept=with_intercept)
            R_midpoint = self.explain(x, first_rule="midpoint", with_intercept=with_intercept)
            #R_midpoint = self.explain(x, first_rule="mean_midpoint", with_intercept=with_intercept)

            R = (1 - eta) * R_GI + eta * R_midpoint

        elif first_rule == "mean_hybrid":
            R_GI = self.explain(x, first_rule="GI", with_intercept=with_intercept)
            R_midpoint = self.explain(x, first_rule="mean_midpoint", with_intercept=with_intercept)
            R = (1 - eta) * R_GI + eta * R_midpoint

        elif first_rule == "hybrid_opposite_SV":

            # make sure that predicted class is not None
            assert predicted_class is not None, "Predicted class must be provided for hybrid_opposite_SV rule"

            R_GI = self.explain(x, first_rule="GI", with_intercept=with_intercept)

            weighted_mean_sv_pos = np.einsum("ni, id -> nd", p_pos, self.x_sup_pos)
            weighted_mean_sv_neg = np.einsum("nj, jd -> nd", p_neg, self.x_sup_neg)

            weighted_mean_sv_pos_sq = np.einsum("ni, id -> nd", p_pos, self.x_sup_pos**2)
            weighted_mean_sv_neg_sq = np.einsum("nj, jd -> nd", p_neg, self.x_sup_neg**2)

            if predicted_class == 1:
                # opp support vector is neg
                corrective_term = weighted_mean_sv_neg_sq - weighted_mean_sv_neg * weighted_mean_sv_pos

            elif predicted_class == -1:
                # opp support vector is pos
                corrective_term = weighted_mean_sv_pos * weighted_mean_sv_neg - weighted_mean_sv_pos_sq
            R = R_GI + 2 * eta * corrective_term

        if reweight_explanation:
            R = self.reweight_explanation(R, x)
        return R

    def reweight_explanation(self, R, x):
        """
        reweight the explanation by scaling it with the neuralised forward.
        :param np.array R: The explanation to be balanced.
        :param np.array x: The input data.
        :return np.array R: The balanced explanation.
        """

        k_pos = rbf_kernel(x, self.x_sup_pos, gamma=self.gamma) * self.alphas_pos[None]
        k_neg = rbf_kernel(x, self.x_sup_neg, gamma=self.gamma) * self.alphas_neg[None]

        g_pos_nb = np.log(k_pos.sum(axis=1))
        g_neg_nb = np.log(k_neg.sum(axis=1))

        assert np.allclose(g_pos_nb - g_neg_nb, self.forward(x))

        if self.intercept_ >0:
            k_pos = np.concatenate([k_pos, np.repeat(self.intercept_, len(x))[:, None]], axis=1)
        elif self.intercept_ < 0:
            k_neg = np.concatenate([k_neg, np.repeat(-self.intercept_, len(x))[:, None]], axis=1)

        g_pos = np.log(k_pos.sum(axis=1))
        g_neg = np.log(k_neg.sum(axis=1))

        R_reweight_pos = np.clip(copy.deepcopy(R), a_min=0, a_max=None)
        R_reweight_neg = np.clip(copy.deepcopy(R), a_max=0, a_min=None)

        R_reweight_pos = R_reweight_pos/(R_reweight_pos.sum(1)[:, None]+1e-9) * g_pos[:, None]
        R_reweight_neg = R_reweight_neg/np.abs((R_reweight_neg.sum(1)[:, None]+1e-9)) * g_neg[:, None]
        R_reweight = R_reweight_pos + R_reweight_neg

        # assert np.allclose(R_balanced.sum(1), (g_pos - g_neg)[:, None])
        return R_reweight

    def closest_pos_SV(self, sample):
        dists = np.linalg.norm(self.x_sup_pos - sample, axis=1)
        return self.x_sup_pos[np.argmin(dists)]

    def closest_neg_SV(self, sample):
        dists = np.linalg.norm(self.x_sup_neg - sample, axis=1)
        return self.x_sup_neg[np.argmin(dists)]

    def closest_midpoint(self, sample):
        pos = self.closest_pos_SV(sample)
        neg = self.closest_neg_SV(sample)
        return (pos + neg) / 2

    def set_gamma(self, gamma):
        self.gamma = gamma