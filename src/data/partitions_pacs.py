def domain_skew_split(domain_indices, n_clients, seed=0):
    """
    Splits the dataset such that each client owns specific domains.
    If n_clients > 4, domains are distributed among clients.
    """
    rng = np.random.default_rng(seed)
    domains = list(domain_indices.keys()) # ['art_painting', 'cartoon', 'photo', 'sketch']
    num_domains = len(domains)
    
    parts = [[] for _ in range(n_clients)]
    
    # Strategy: Assign domains to clients
    # If clients = 4, each gets 1.
    # If clients = 5, first 4 get 1, 5th gets a mix or a specific one.
    
    for i in range(n_clients):
        # Determine which domain this client gets
        # Using modulo allows cycling if clients > domains
        target_domain = domains[i % num_domains] 
        
        # Get all indices for that domain
        indices = domain_indices[target_domain]
        
        # If multiple clients share a domain (e.g. 8 clients, 4 domains),
        # we need to split that domain's indices between them.
        # Simple approach here: Each client gets a random subset of that domain 
        # (or the whole domain if you want overlaps).
        
        # Let's do distinct subsets (no overlap between clients)
        # Check how many clients are assigned to this domain
        clients_sharing_domain = [c for c in range(n_clients) if domains[c % num_domains] == target_domain]
        
        # Shuffle indices of the domain
        rng.shuffle(indices)
        
        # Split indices among those clients
        n_sharing = len(clients_sharing_domain)
        chunk_size = len(indices) // n_sharing
        
        # Find which "chunk" this client (i) should take
        rank = clients_sharing_domain.index(i)
        start = rank * chunk_size
        end = start + chunk_size if rank < n_sharing - 1 else len(indices)
        
        parts[i] = indices[start:end]
        
        print(f"Client {i} assigned to Domain: {target_domain} (n={len(parts[i])})")
        
    return parts


def split_client_train_test_strict(idxs, full_dataset, test_frac, seed, num_classes):
    """
    Stratified per client:
    - For this client's indices only (idxs),
    - split per class into train/test with the SAME counts per class 
      across all clients (because each client started from iid_split_strict).
    """
    rng = np.random.default_rng(seed)
    labels = np.array(full_dataset.targets)

    train_idx = []
    test_idx  = []

    for cls in range(num_classes):
        # all this client's indices that belong to class cls
        cls_idx = [i for i in idxs if labels[i] == cls]
        cls_idx = np.array(cls_idx, dtype=int)

        # shuffle only within this class
        rng.shuffle(cls_idx)

        n_c = len(cls_idx)
        n_test_c = int(n_c * test_frac)   # same for all clients since n_c is the same
        n_train_c = n_c - n_test_c

        test_idx.extend(cls_idx[:n_test_c])
        train_idx.extend(cls_idx[n_test_c:])

    return np.array(train_idx, dtype=int), np.array(test_idx, dtype=int)


def make_fewshot(train_idx, full_dataset, shots_per_class):
    """
    Returns only up to 'shots_per_class' samples per class 
    from this client's training indices.
    """
    labels = np.array(full_dataset.targets)

    selected = []
    for cls in range(10):        # CIFAR-10 has 10 classes
        cls_idx = [i for i in train_idx if labels[i] == cls]
        np.random.shuffle(cls_idx)
        selected.extend(cls_idx[:shots_per_class])  # take first K

    return np.array(selected)
