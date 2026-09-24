import numpy as np

def score_msp(logits):
    # u_MSP(x) = 1 - max p_k(x)
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return 1.0 - np.max(probs, axis=1)

def score_mls(logits):
    # u_MLS(x) = -max z_k(x)
    return -np.max(logits, axis=1)

def score_energy(logits):
    # u_Energy(x) = -log \sum exp(z_k(x))
    # logsumexp trick
    max_l = np.max(logits, axis=1, keepdims=True)
    lse = max_l + np.log(np.sum(np.exp(logits - max_l), axis=1, keepdims=True))
    return -lse.squeeze()

def fit_mahalanobis(train_features, train_labels, num_classes=10):
    means = []
    covariances = []
    for c in range(num_classes):
        feats_c = train_features[train_labels == c]
        mu = np.mean(feats_c, axis=0)
        means.append(mu)
        cov = np.cov(feats_c.T)
        covariances.append(cov)
        
    means = np.array(means)
    shared_cov = np.mean(covariances, axis=0)
    # add 1e-6 to diagonal
    shared_cov += np.eye(shared_cov.shape[0]) * 1e-6
    inv_cov = np.linalg.inv(shared_cov)
    return means, inv_cov

def score_mahalanobis(features, means, inv_cov):
    # min_c (f(x) - \mu_c)^T \Sigma^{-1} (f(x) - \mu_c)
    n = features.shape[0]
    num_classes = means.shape[0]
    dists = np.zeros((n, num_classes))
    for c in range(num_classes):
        diff = features - means[c]
        # (N, D) x (D, D) x (D, N) -> diag gives (N,)
        dist = np.sum(np.dot(diff, inv_cov) * diff, axis=1)
        dists[:, c] = dist
    return np.min(dists, axis=1)

def score_proser_placeholder(logits):
    # logits shape: (N, 15) -> 10 known, 5 dummy
    # We combine the strongest dummy response with the known-class responses
    max_known = np.max(logits[:, :10], axis=1)
    max_dummy = np.max(logits[:, 10:], axis=1)
    # Larger values indicate greater novelty
    return max_dummy - max_known
