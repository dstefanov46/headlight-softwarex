import torch

import numpy as np

from .adj_mx import get_adj


class GCNLayer(torch.nn.Module):
    """
    One Graph Convolutional Network (GCN) layer.

    Args:
        input_size (int): Number of input features per node.
        out_feature (int): Number of output features.
        adj_num (int): Number of adjacency matrices used.
        with_res (bool): Whether to use a residual connection.

    Shape:
        - Input:
            x: (batch_size, num_nodes, input_size)
            adj: (adj_num, num_nodes, num_nodes) or (num_nodes, num_nodes)
        - Output:
            Tensor: (batch_size, num_nodes, out_feature)
    """
    def __init__(self, input_size, out_feature, adj_num, with_res=True):
        super(GCNLayer, self).__init__()
        self.with_res = with_res
        self.adj_num = adj_num

        self.w = torch.nn.Linear(adj_num * input_size, out_feature, bias=False)

        if with_res:
            self.w_res = torch.nn.Linear(out_feature, input_size)

    def forward(self, x, adj):
        """Forward pass of GCNLayer."""
        b, node_num, _ = x.size()
        if self.adj_num == 1:
            y = torch.matmul(adj, x)
        else:
            x1 = x.unsqueeze(dim=-2)
            y = torch.matmul(adj, x1).transpose(-2, -3).reshape(b, node_num, -1)
            y = torch.relu(self.w(y))
        if self.with_res:
            y = self.w_res(y) + x
        return y


class GCN(torch.nn.Module):
    """
    Multi-layer Graph Convolutional Network (GCN).

    Args:
        input_size (int): Dimension of input features.
        hidden_sizes (list of int): Hidden layer sizes.
        step_num_out (int): Output feature dimension.
        adj_path (str): Path to the adjacency matrix file.
        device (str): Device to run the model on (e.g., 'cpu' or 'cuda').
        adj_type (str): Type of adjacency normalization ('gcn' or 'cheb').
        with_res (bool): Whether to use residual connections.
        **kwargs: Additional arguments (e.g., K for Chebyshev order).

    Note:
        If adj_type is 'cheb', 'K' controls the polynomial order.
    """
    def __init__(self, input_size, hidden_sizes, step_num_out, adj_path, device,
                 adj_type='gcn', with_res=True, **kwargs):
        super(GCN, self).__init__()

        device = torch.device(device if torch.cuda.is_available() else 'cpu')
        if adj_type == 'cheb':
            self.K = kwargs.get('K', 3)
        else:
            self.K = 1

        self.adj_mx = self.build_adj_matrix(np.load(adj_path), device, adj_type=adj_type, **kwargs)
        self.gcn_list = self._build_gcn_network(input_size, hidden_sizes, step_num_out, with_res)

    def _build_gcn_network(self, input_size, hidden_sizes, step_num_out, with_res=True):
        """Build GCN network with specified layers."""
        gcn_list = torch.nn.ModuleList()

        f_in = input_size
        for hidden_size in hidden_sizes:
            gcn_list.append(GCNLayer(f_in, hidden_size, self.K, with_res))
            if not with_res:
                f_in = hidden_size
        gcn_list.append(GCNLayer(f_in, step_num_out, self.K, with_res=False))

        return gcn_list

    @staticmethod
    def build_adj_matrix(adj, device, adj_type=None, **kwargs):
        """
        Build (possibly Chebyshev-expanded) adjacency matrix tensor.

        Args:
            adj (ndarray): Adjacency matrix.
            device (torch.device): Target device.
            adj_type (str): Type of adjacency ('gcn' or 'cheb').
            **kwargs: May include 'K' for Chebyshev order.

        Returns:
            Tensor: Adjacency matrix or stack of Chebyshev polynomials.
        """
        K = kwargs.get('K', 3)
        adj_mx = get_adj(adj, adj_type=adj_type)

        if adj_type != 'cheb':
            return torch.tensor(adj_mx, dtype=torch.float32, device=device)

        cheb_list = [np.eye(adj_mx.shape[0]), adj_mx]
        for _ in range(2, K):
            cheb_list.append(2 * np.dot(adj_mx, cheb_list[-1]) - cheb_list[-2])

        return torch.tensor(cheb_list, dtype=torch.float32, device=device)

    def forward(self, x):
        """
        Forward pass through the full GCN model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, num_nodes, input_size).

        Returns:
            Tensor: Output tensor of shape (batch_size, num_nodes, step_num_out)
        """
        for gcn_layer in self.gcn_list:
            x = gcn_layer(x, self.adj_mx)
        return x.squeeze()
