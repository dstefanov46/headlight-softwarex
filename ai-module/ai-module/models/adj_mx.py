import numpy as np


def normalized_laplacian(adj):
    """
    Compute the normalized Laplacian matrix.

    Formula:
        I - D^(-1/2) * A * D^(-1/2)

    Args:
        adj (np.ndarray): Adjacency matrix.

    Returns:
        np.ndarray: Normalized Laplacian matrix.
    """
    N = adj.shape[0]
    I = np.eye(N)
    d = adj.sum(axis=-1) ** (-0.5)
    d[np.isinf(d)] = 0.
    return I - d.reshape(-1, 1) * adj * d


def laplacian(adj):
    """
    Compute the unnormalized Laplacian matrix.

    Formula:
        D - A

    Args:
        adj (np.ndarray): Adjacency matrix.

    Returns:
        np.ndarray: Laplacian matrix.
    """
    d = np.diag(adj.sum(axis=-1))
    return d - adj


def gcn(adj):
    """
    Compute the GCN-normalized adjacency matrix.

    Formula:
        D^(-1/2) * (A + I) * D^(-1/2)

    Args:
        adj (np.ndarray): Adjacency matrix.

    Returns:
        np.ndarray: GCN-normalized adjacency matrix.
    """
    adj = random_walk(adj)
    N = adj.shape[0]
    adj += np.eye(N)
    d = adj.sum(axis=-1) ** (-0.5)
    d[np.isinf(d)] = 0.
    return d.reshape(-1, 1) * adj * d


def random_walk_laplacian(adj):
    """
    Compute the random walk Laplacian.

    Formula:
        I - D^(-1) * A

    Args:
        adj (np.ndarray): Adjacency matrix.

    Returns:
        np.ndarray: Random walk Laplacian matrix.
    """
    d = adj.sum(axis=-1) ** (-1)
    d[np.isinf(d)] = 0.
    return np.eye(adj.shape[0]) - d.reshape(-1, 1) * adj


def random_walk(adj):
    """
    Compute the random walk normalized adjacency matrix.

    Formula:
        D^(-1) * A

    Args:
        adj (np.ndarray): Adjacency matrix.

    Returns:
        np.ndarray: Random walk normalized adjacency.
    """
    d = adj.sum(axis=-1) ** (-1)
    d[np.isinf(d)] = 0.
    return d.reshape(-1, 1) * adj


def cheb(adj):
    """
    Compute the Chebyshev-scaled Laplacian.

    Formula:
        (2 / lambda_max) * L - I,
        where L = I - D^(-1/2) * A * D^(-1/2)

    Args:
        adj (np.ndarray): Adjacency matrix.

    Returns:
        np.ndarray: Chebyshev-scaled Laplacian matrix.
    """
    N = adj.shape[0]
    I = np.eye(N)
    L = normalized_laplacian(adj)
    max_eig_value = abs(np.linalg.eigvals(L)).max()
    return 2 / max_eig_value * L - I


def get_adj(adj, adj_type):
    """
    Retrieve a specific type of adjacency matrix transformation.

    Args:
        adj (np.ndarray): Adjacency matrix.
        adj_type (str): Type of transformation. One of:
            ['normalized_laplacian', 'laplacian', 'gcn',
             'random_walk_laplacian', 'random_walk', 'cheb']

    Returns:
        np.ndarray: Transformed adjacency matrix.
    """
    adj_types = {
        'normalized_laplacian': normalized_laplacian,
        'laplacian': laplacian,
        'gcn': gcn,
        'random_walk_laplacian': random_walk_laplacian,
        'random_walk': random_walk,
        'cheb': cheb
    }
    func = adj_types.get(adj_type, lambda: "Invalid adj_type!")
    return func(adj)
