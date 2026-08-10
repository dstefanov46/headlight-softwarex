import math
import torch

import numpy as np
import torch.nn.functional as F

from .GCN import GCN


class SpatialAttention(torch.nn.Module):
    """
    Compute spatial attention scores between graph nodes.

    Args:
        num_nodes (int): Number of nodes.
        f_in (int): Number of features.
        c_in (int): Number of time steps.

    Shape:
        - Input: (batch_size, c_in, num_nodes, f_in)
        - Output: (batch_size, num_nodes, num_nodes)
    """
    def __init__(self, num_nodes, f_in, c_in):
        super(SpatialAttention, self).__init__()
        self.w1 = torch.nn.Conv2d(c_in, 1, 1, bias=False)
        self.w2 = torch.nn.Linear(f_in, c_in, bias=False)
        self.w3 = torch.nn.Parameter(torch.randn(f_in, dtype=torch.float32), requires_grad=True)
        self.vs = torch.nn.Parameter(torch.randn(num_nodes, num_nodes, dtype=torch.float32), requires_grad=True)
        self.bs = torch.nn.Parameter(torch.randn(num_nodes, num_nodes, dtype=torch.float32), requires_grad=True)

        torch.nn.init.kaiming_uniform_(self.vs, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.bs, a=math.sqrt(5))
        torch.nn.init.uniform_(self.w3)

    def forward(self, x):
        y1 = self.w2(self.w1(x).squeeze(dim=1))
        y2 = torch.matmul(x, self.w3)
        product = torch.matmul(y1, y2)
        y = torch.matmul(self.vs, torch.sigmoid(product + self.bs))
        return F.softmax(y, dim=-1)


class ChebConv(torch.nn.Module):
    """
    Graph Convolution with Chebyshev polynomials.

    Args:
        input_feature (int): Dimension of input features.
        out_feature (int): Dimension of output features.
        adj_mx (Tensor): Precomputed Chebyshev polynomials of the adjacency matrix with shape (K, num_nodes, num_nodes).

    Shape:
        - Input: x (batch_size, c_in, num_nodes, f_in), spatial_att (batch_size, num_nodes, num_nodes)
        - Output: (batch_size, c_in, num_nodes, out_feature)
    """
    def __init__(self, input_feature, out_feature, adj_mx):
        super(ChebConv, self).__init__()
        self.adj_mx = adj_mx
        self.w = torch.nn.Linear(adj_mx.size(0) * input_feature, out_feature, bias=False)

    def forward(self, x, spatial_att):
        b, c_in, num_nodes, _ = x.size()
        outputs = []
        adj = spatial_att.unsqueeze(dim=1) * self.adj_mx
        for i in range(c_in):
            x1 = x[:, i].unsqueeze(dim=1)
            y = torch.matmul(adj, x1).transpose(1, 2).reshape(b, num_nodes, -1)
            y = torch.relu(self.w(y))
            outputs.append(y)
        return torch.stack(outputs, dim=1)


class TemporalAttention(torch.nn.Module):
    """
    Compute temporal attention scores between time steps.

    Args:
        num_nodes (int): Number of nodes.
        f_in (int): Number of features.
        c_in (int): Number of time steps.

    Shape:
        - Input: (batch_size, c_in, num_nodes, f_in)
        - Output: (batch_size, c_in, c_in)
    """
    def __init__(self, num_nodes, f_in, c_in):
        super(TemporalAttention, self).__init__()
        self.w1 = torch.nn.Parameter(torch.randn(num_nodes, dtype=torch.float32), requires_grad=True)
        self.w2 = torch.nn.Linear(f_in, num_nodes, bias=False)
        self.w3 = torch.nn.Parameter(torch.randn(f_in, dtype=torch.float32), requires_grad=True)
        self.be = torch.nn.Parameter(torch.randn(1, c_in, c_in, dtype=torch.float32), requires_grad=True)
        self.ve = torch.nn.Parameter(torch.zeros(c_in, c_in, dtype=torch.float32), requires_grad=True)

        torch.nn.init.kaiming_uniform_(self.ve, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.be, a=math.sqrt(5))
        torch.nn.init.uniform_(self.w1)
        torch.nn.init.uniform_(self.w3)

    def forward(self, x):
        y1 = self.w2(torch.matmul(x.transpose(2, 3), self.w1))
        y2 = torch.matmul(x, self.w3).transpose(1, 2)
        product = torch.matmul(y1, y2)
        E = torch.matmul(self.ve, torch.sigmoid(product + self.be))
        return F.softmax(E, dim=-1)


class ASTGCNBlock(torch.nn.Module):
    """
    One ASTGCN block that includes spatial and temporal attention, Chebyshev graph convolution, and temporal
    convolution.

    Args:
        c_in (int): Number of time steps.
        f_in (int): Number of input features.
        num_cheb_filter (int): Number of Chebyshev filters.
        num_time_filter (int): Number of filters in temporal convolution.
        kernel_size (int): Kernel size of temporal convolution.
        adj_mx (Tensor): Adjacency matrix.
        stride (int): Stride for temporal convolution.
        padding (int): Padding for temporal convolution.

    Shape:
        - Input: (batch_size, c_in, num_nodes, f_in)
        - Output: (batch_size, c_out, num_nodes, num_time_filter)
    """
    def __init__(self, c_in, f_in, num_cheb_filter, num_time_filter, kernel_size, adj_mx,
                 stride=1, padding=0):
        super(ASTGCNBlock, self).__init__()
        self.adj_mx = adj_mx
        self._c_out_conv = (c_in + 2 * padding - kernel_size) // stride + 1
        _c_out_res = (c_in - 1) // stride + 1
        self._res_start = (_c_out_res - self._c_out_conv) // 2

        num_nodes = adj_mx.size(-1)
        self.spatial_att = SpatialAttention(num_nodes, f_in, c_in)
        self.cheb_conv = ChebConv(f_in, num_cheb_filter, self.adj_mx)
        self.temporal_att = TemporalAttention(num_nodes, f_in, c_in)

        self.time_conv = torch.nn.Conv2d(
            in_channels=num_cheb_filter,
            out_channels=num_time_filter,
            kernel_size=(1, kernel_size),
            stride=(1, stride),
            padding=(0, padding)
        )

        self.residual_conv = torch.nn.Conv2d(
            in_channels=f_in,
            out_channels=num_time_filter,
            kernel_size=(1, 1),
            stride=(1, stride)
        )

        self.ln = torch.nn.LayerNorm(num_time_filter)

    def forward(self, x):
        b, c, n_d, f = x.size()
        temporal_att = self.temporal_att(x)
        x_tat = torch.matmul(temporal_att, x.reshape(b, c, -1)).reshape(b, c, n_d, f)
        spatial_att = self.spatial_att(x_tat)
        spatial_gcn = self.cheb_conv(x, spatial_att)

        time_conv_output = self.time_conv(spatial_gcn.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        x_residual = self.residual_conv(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        x_residual = x_residual[:, self._res_start: self._res_start + self._c_out_conv]

        return self.ln(F.relu(x_residual + time_conv_output))


class ASTGCN(torch.nn.Module):
    """
    Spatio-temporal graph convolutional network using adaptive spatial and temporal attention.

    Args:
        num_block (int): Number of ASTGCN blocks.
        step_num_in (int): Number of input time steps.
        step_num_out (int): Number of output time steps.
        input_size (int): Number of input features.
        num_cheb_filter (int): Number of Chebyshev filters.
        num_time_filter (int): Number of filters in temporal convolution.
        kernel_size (int): Kernel size of temporal convolution.
        stride (int): Stride for temporal convolution.
        padding (int): Padding for temporal convolution.
        K (int): Order of Chebyshev polynomials.
        adj_path (str): Path to the adjacency matrix.
        device (str): Target device (e.g., 'cuda' or 'cpu').
    """
    def __init__(self, **kwargs):
        super(ASTGCN, self).__init__()

        num_block = kwargs.get('num_block', 2)
        c_in = kwargs.get('step_num_in', 12)
        c_out = kwargs.get('step_num_out', 12)
        f_in = kwargs.get('input_size', 1)
        num_cheb_filter = kwargs.get('num_cheb_filter', 64)
        num_time_filter = kwargs.get('num_time_filter', 64)
        kernel_size = kwargs.get('kernel_size', 3)
        stride = kwargs.get('stride', 1)
        padding = kwargs.get('padding', 0)
        K = kwargs.get('K', 2)
        adj_mx_path = kwargs.get('adj_path', None)
        device_name = kwargs.get('device', None)
        device = torch.device(device_name if torch.cuda.is_available() else 'cpu')

        if adj_mx_path is None:
            raise ValueError('Please set the path of adjacent matrix !')

        adj = np.load(adj_mx_path)
        adj_mx = GCN.build_adj_matrix(adj, device, adj_type='cheb', K=K)

        self.blocks = torch.nn.Sequential()
        for i in range(1, num_block + 1):
            self.blocks.add_module(
                name='block_{}'.format(i),
                module=ASTGCNBlock(c_in, f_in, num_cheb_filter, num_time_filter,
                                   kernel_size, adj_mx, stride, padding)
            )
            c_in, f_in = (c_in + 2 * padding - kernel_size) // stride + 1, num_time_filter

        self.final_conv = torch.nn.Conv2d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=(1, num_time_filter)
        )

    def forward(self, x):
        y = self.blocks(x)
        return self.final_conv(y).squeeze()
