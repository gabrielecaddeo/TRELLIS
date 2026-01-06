import numpy as np

def feature_spread_stats(feats, n_pairs=20000, seed=0):
    rng = np.random.default_rng(seed)
    N = feats.shape[0]
    # norms
    norms = np.linalg.norm(feats, axis=1)
    # random pair cosine
    i = rng.integers(0, N, size=n_pairs)
    j = rng.integers(0, N, size=n_pairs)
    a = feats[i]; b = feats[j]
    cos = (a*b).sum(1) / (np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1) + 1e-8)
    return {
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "cos_rand_mean": float(cos.mean()),
        "cos_rand_p10": float(np.percentile(cos,10)),
        "cos_rand_p90": float(np.percentile(cos,90)),
    }

# usage
pack = np.load("datasets/HSSD/features/dinov2_vitl14_reg/004e1a8b3146a0e980f701c0c25bc58b3e653d238d96a586b69be44e6f0d6e71_f000.npz")
feats = pack["patchtokens"].astype(np.float32)
print(feature_spread_stats(feats))
