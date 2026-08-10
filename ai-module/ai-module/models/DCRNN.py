import torch

import numpy as np

from .adj_mx import get_adj


def _seed_compatible_uniform() -> float:
    """Draw through Generator while preserving NumPy's seeded legacy stream."""
    legacy_state = np.random.get_state()
    bit_generator = np.random.MT19937(seed=0)
    generator_state = bit_generator.state
    generator_state["state"]["key"] = legacy_state[1].copy()
    generator_state["state"]["pos"] = legacy_state[2]
    bit_generator.state = generator_state
    value = float(np.random.Generator(bit_generator).uniform(0, 1))
    next_state = bit_generator.state["state"]
    np.random.set_state(
        (
            "MT19937",
            next_state["key"],
            next_state["pos"],
            legacy_state[3],
            legacy_state[4],
        )
    )
    return value


class DCGRUCell(torch.nn.Module):
    """
    A GRU cell extended with graph convolution.

    Args:
        input_size (int): Size of input features.
        rnn_hidden_size (int): Number of hidden units.
        use_gc (bool): Whether to use graph convolution.
        activation (str): Name of activation function.
        **kwargs: Must include 'graph_num' if use_gc=True. May include 'K' for Chebyshev order.
    """
    def __init__(self, input_size, rnn_hidden_size, use_gc=True, activation='tanh', **kwargs):
        super(DCGRUCell, self).__init__()
        self.in_size = input_size
        self.h_size = rnn_hidden_size
        self.use_gc = use_gc
        self.activation = getattr(torch, activation, torch.tanh)

        if use_gc:
            assert 'graph_num' in kwargs, 'graph_num is not set !'
            self.graph_num = kwargs.get('graph_num')
            self.K = kwargs.get('K', 3)

        gw_size = self.in_size + self.h_size
        if use_gc:
            gw_size *= self.graph_num * (self.K - 1) + 1

        self.gate_weight = torch.nn.Linear(gw_size, self.h_size * 2)
        self.cell_weight = torch.nn.Linear(gw_size, self.h_size)

    def _gconv(self, adj_mx_list, x):
        """
        Apply graph convolution using Chebyshev polynomials.

        Args:
            adj_mx_list (list): List of adjacency matrices.
            x (Tensor): Input of shape (batch_size, num_nodes, features).

        Returns:
            Tensor: Graph convolved output.
        """
        h = [x]
        for adj_mx in adj_mx_list:
            tk_2, tk_1 = x, torch.matmul(adj_mx, x)
            h.append(tk_1)
            for _ in range(2, self.K):
                tk_2, tk_1 = tk_1, 2 * torch.matmul(adj_mx, tk_1) - tk_2
                h.append(tk_1)
        return torch.cat(h, dim=-1)

    def forward(self, x, hx, adj_mx_list=None):
        """
        Forward pass of the DCGRU cell.

        Args:
            x (Tensor): Input tensor.
            hx (Tensor): Hidden state.
            adj_mx_list (list): List of adjacency matrices.

        Returns:
            Tensor: Updated hidden state.
        """
        h = torch.cat((x, hx), dim=-1)
        if self.use_gc:
            h = self._gconv(adj_mx_list, h)
        h = torch.sigmoid(self.gate_weight(h))
        r, z = torch.split(tensor=h, split_size_or_sections=self.h_size, dim=-1)

        c = torch.cat((x, r * hx), dim=-1)
        if self.use_gc:
            c = self._gconv(adj_mx_list, c)
        c = self.activation(self.cell_weight(c))

        hx = z * hx + (1 - z) * c
        return hx


class DCGRULayer(torch.nn.Module):
    """
    Multi-layer GRU with graph convolution.

    Args:
        input_size (int): Input feature dimension.
        rnn_hidden_size (int): Hidden state size.
        rnn_layer_num (int): Number of stacked GRU layers.
        use_gc (bool): Use graph convolution.
    """
    def __init__(self, input_size, rnn_hidden_size, rnn_layer_num, use_gc=True, **kwargs):
        super(DCGRULayer, self).__init__()
        self.h_size = rnn_hidden_size
        self.dcgru_layers = torch.nn.ModuleList(
            [DCGRUCell(input_size, rnn_hidden_size, use_gc, **kwargs)]
        )
        for _ in range(1, rnn_layer_num):
            self.dcgru_layers.append(
                DCGRUCell(rnn_hidden_size, rnn_hidden_size, use_gc, **kwargs)
            )

    def forward(self, x, hx, adj_mx_list):
        """
        Forward pass for multi-layer GRU.

        Args:
            x (Tensor): Input tensor.
            hx (Tensor): Initial hidden states.
            adj_mx_list (list): List of adjacency matrices.

        Returns:
            Tuple[Tensor, Tensor]: Output and stacked hidden states.
        """
        hidden_states = []
        h = x
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            h = dcgru_layer(h, hx[layer_num], adj_mx_list)
            hidden_states.append(h)
        return h, torch.stack(hidden_states)


class GeneralDCRNN(torch.nn.Module):
    """
    General DCRNN model for spatiotemporal prediction.

    Supports encoder-decoder structure with optional curriculum learning.
    """
    def __init__(self, **kwargs):
        super(GeneralDCRNN, self).__init__()

        self.input_size = kwargs.get('input_size', 1)
        self.output_size = 1
        self.rnn_hidden_size = kwargs.get('rnn_hidden_size', 32)
        self.rnn_layer_num = kwargs.get('rnn_layer_num', 1)
        self.use_gc = kwargs.get('use_gc', True)
        self.step_num_out = kwargs.get('step_num_out', 12)
        self.batch_first = kwargs.get('batch_first', False)
        self.cl_decay_steps = kwargs.get('cl_decay_steps', 2000)
        self.use_curriculum_lr = kwargs.get('use_curriculum_learning', True)
        self.activation = kwargs.get('activation', 'tanh')
        device_name = kwargs.get('device', 'cuda:0')
        self.device = torch.device(device_name if torch.cuda.is_available() else 'cpu')

        K = kwargs.get('K', 3)
        graph_num = kwargs.get('graph_num', 2)
        cell_kwargs = {'graph_num': graph_num, 'K': K} if self.use_gc else {}

        self.encoder_layer = DCGRULayer(
            input_size=self.input_size,
            rnn_hidden_size=self.rnn_hidden_size,
            rnn_layer_num=self.rnn_layer_num,
            use_gc=self.use_gc,
            activation=self.activation,
            **cell_kwargs
        )
        self.decoder_layer = DCGRULayer(
            input_size=self.output_size,
            rnn_hidden_size=self.rnn_hidden_size,
            rnn_layer_num=self.rnn_layer_num,
            use_gc=self.use_gc,
            activation=self.activation,
            **cell_kwargs
        )

        self.fc_layers = torch.nn.ModuleList([
            torch.nn.Linear(self.rnn_hidden_size, self.output_size) for _ in range(self.step_num_out)
        ])

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def encoder(self, x, adj_mx_list=None):
        """
        Encode input sequence over time.

        Args:
            x (Tensor): Input tensor.
            adj_mx_list (list): List of adjacency matrices.

        Returns:
            Tensor: Hidden states after encoding.
        """
        if self.batch_first:
            x = x.transpose(0, 1)

        h_size = [self.rnn_layer_num] + list(x.size()[1: -1]) + [self.rnn_hidden_size]
        hx = torch.zeros(h_size, dtype=torch.float32, device=self.device)

        for t in range(x.size(0)):
            _, hx = self.encoder_layer(x[t], hx, adj_mx_list)

        return hx

    def decoder(self, hx, adj_mx_list=None, labels=None, batches_seen=None):
        """
        Decode future sequence using the encoded state.

        Args:
            hx (Tensor): Hidden state from encoder.
            adj_mx_list (list): List of adjacency matrices.
            labels (Tensor): Ground truth labels (optional).
            batches_seen (int): Training step counter (for curriculum learning).

        Returns:
            Tensor: Predicted sequence.
        """
        x_size = list(hx.size()[1: -1]) + [self.output_size]
        x = torch.zeros(x_size, dtype=torch.float32, device=self.device)

        if labels is not None:
            labels = labels.unsqueeze(dim=-1)
            if self.step_num_out > 1 and self.batch_first:
                labels = labels.transpose(0, 1)
            elif self.step_num_out == 1:
                labels = labels.unsqueeze(dim=0)

        y = []
        for t in range(self.step_num_out):
            x, hx = self.decoder_layer(x, hx, adj_mx_list)
            x = self.fc_layers[t](x)
            y.append(x)
            if self.training and self.use_curriculum_lr:
                c = _seed_compatible_uniform()
                if c < self._compute_sampling_threshold(batches_seen):
                    x = labels[t]

        y = torch.stack(y).squeeze(dim=-1)
        if self.batch_first and self.step_num_out >= 1:
            y = y.transpose(0, 1)

        return y

    def forward(self, x, adj_mx_list, labels=None, batches_seen=None):
        """
        Full forward pass of the DCRNN model.

        Args:
            x (Tensor): Input sequence.
            adj_mx_list (list): List of adjacency matrices.
            labels (Tensor): Ground truth labels.
            batches_seen (int): Number of batches seen in training.

        Returns:
            Tensor: Output prediction.
        """
        encoder_hidden_state = self.encoder(x, adj_mx_list)
        outputs = self.decoder(
            encoder_hidden_state,
            adj_mx_list,
            labels=labels,
            batches_seen=batches_seen,
        )
        return outputs


class DCRNN(GeneralDCRNN):
    """
    DCRNN model that loads and applies multiple adjacency matrices.
    """
    def __init__(self, **kwargs):
        super(DCRNN, self).__init__(**kwargs)
        adj_path = kwargs.get('adj_path', None)
        adj_type = kwargs.get('adj_type', None)
        if adj_path is None:
            raise ValueError('The path of adjacent matrix should be offered!')
        self.adj_mx_list = self._load_adj(adj_path, adj_type, self.device)

    @staticmethod
    def _load_adj(adj_path: str, adj_type: str, device, **kwargs):
        """
        Load adjacency matrices from path.

        Args:
            adj_path (str): Path to the adjacency matrix file(s).
            adj_type (str): Type of adjacency processing.
            device (torch.device): Target device.

        Returns:
            list: List of adjacency matrices.
        """
        adj_paths = adj_path.strip().split(',')
        adj_mx_list = []
        for adj_mx_path in adj_paths:
            adj = np.load(adj_mx_path.strip())
            adj_mx = torch.tensor(get_adj(adj, adj_type), dtype=torch.float32, device=device)
            adj_mx_list.append(adj_mx)
            adj_mx = torch.tensor(get_adj(adj.T, adj_type), dtype=torch.float32, device=device)
            adj_mx_list.append(adj_mx)
        return adj_mx_list

    def forward(self, x, adj_mx_list=None, labels=None, batches_seen=None):
        """
        Forward pass with loaded adjacency matrices.

        Args:
            x (Tensor): Input sequence.
            adj_mx_list (list): Accepted for compatibility with GeneralDCRNN;
                this model always uses its loaded adjacency matrices.
            labels (Tensor): Ground truth labels.
            batches_seen (int): Number of batches seen in training.

        Returns:
            Tensor: Output prediction.
        """
        loaded_adj_mx_list = self.adj_mx_list
        return super().forward(
            x,
            loaded_adj_mx_list,
            labels=labels,
            batches_seen=batches_seen,
        )
