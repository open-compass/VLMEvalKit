import numpy as np


def ransac(
    data,
    model_class,
    min_samples,
    residual_threshold,
    max_trials=10000,
):
    src, dst = data
    if src.shape[0] != dst.shape[0]:
        raise ValueError("src and dst must have the same number of samples")
    if src.shape[0] < min_samples:
        raise ValueError("not enough data points to estimate model")
    random_state=2026
    rng = np.random.default_rng(random_state)
    indices = np.arange(src.shape[0])

    best_model = None
    best_inliers = None
    best_inlier_count = 0
    best_residual_sum = np.inf

    for _ in range(max_trials):
        sample_idx = rng.choice(indices, size=min_samples, replace=False)
        model = model_class()
        try:
            ok = model.estimate(src[sample_idx], dst[sample_idx])
        except Exception:
            continue
        if ok is False:
            continue

        residuals = model.residuals(src, dst)
        if residuals is None:
            continue
        residuals = np.asarray(residuals)
        inliers = residuals < residual_threshold
        inlier_count = int(np.count_nonzero(inliers))
        if inlier_count == 0:
            continue
        residual_sum = float(np.sum(residuals[inliers]))

        if (
            inlier_count > best_inlier_count
            or (inlier_count == best_inlier_count and residual_sum < best_residual_sum)
        ):
            best_model = model
            best_inliers = inliers
            best_inlier_count = inlier_count
            best_residual_sum = residual_sum

    if best_inliers is None:
        return None, None

    refined_model = model_class()
    try:
        ok = refined_model.estimate(src[best_inliers], dst[best_inliers])
    except Exception:
        return best_model, best_inliers
    if ok is False:
        return best_model, best_inliers
    return refined_model, best_inliers
