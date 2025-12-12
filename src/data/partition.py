import numpy as np
from collections import Counter

def quantity_and_label_skew_split(
    trainset,
    k,
    alpha_qty,
    alpha_label,
    min_per_client,
    seed,
    max_tries,
):
    """
    Combined quantity + label skew using Dirichlet.

    - alpha_qty   controls how unbalanced total client sizes are
    - alpha_label controls how skewed class proportions are per client
    - min_per_client is a soft constraint; we resample until everyone has at least this many points

    Returns:
        parts: list of length k, each an np.array of indices into trainset
    """
    rng = np.random.default_rng(cfg.seed)
    targets = np.array(trainset.targets)
    N = len(targets)

    classes = np.unique(targets)
    C = len(classes)

    # Indices per class (e.g., CIFAR-10 -> 10 classes, each ~6000 samples)
    idx_by_class = {c: np.where(targets == c)[0].tolist() for c in classes}

    # Try a few times until everyone has >= min_per_client samples
    for _ in range(max_tries):
        # fresh shuffled pools each attempt
        pools = {c: list(idx_by_class[c]) for c in classes}
        for c in classes:
            rng.shuffle(pools[c])

        parts = [[] for _ in range(k)]

        # ▶ 1) Sample "size tendencies" for clients (quantity skew)
        #     Clients with bigger size_props will tend to get more samples from every class
        size_props = rng.dirichlet([alpha_qty] * k)  # shape (k,), sums to 1

        # ▶ 2) For each class, allocate its samples using a Dirichlet
        #      whose parameters are biased by size_props (so big clients get more of every class)
        for c in classes:
            pool = pools[c]
            Nc = len(pool)

            # Dirichlet concentration for this class, biased by size_props
            base_conc = alpha_label * k
            params = base_conc * size_props           # shape (k,)

            p = rng.dirichlet(params)                 # class-c proportions per client
            raw = p * Nc
            base = np.floor(raw).astype(int)
            deficit = Nc - base.sum()

            if deficit > 0:
                frac = raw - base
                bump = np.argsort(-frac)[:deficit]
                base[bump] += 1

            # Now sum_k base[k] == Nc
            start = 0
            for i in range(k):
                take = int(base[i])
                if take > 0:
                    parts[i].extend(pool[start:start + take])
                    start += take

        # Shuffle inside each client and check sizes
        parts = [np.array(rng.permutation(p), dtype=int) for p in parts]
        sizes = [len(p) for p in parts]

        if min(sizes) >= min_per_client:
            # Success: everyone has enough data
            return parts

    raise RuntimeError("Failed to sample a split satisfying min_per_client within max_tries")
