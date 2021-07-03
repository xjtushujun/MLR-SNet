import torch
import copy
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList, _VF
from module import MetaModule, MetaLinear, to_var

class MetaEmbedding(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Embedding(*args, **kwargs)

        self.num_embeddings = ignore.num_embeddings
        self.embedding_dim = ignore.embedding_dim
        self.padding_idx = ignore.padding_idx
        self.max_norm = ignore.max_norm
        self.norm_type = ignore.norm_type
        self.scale_grad_by_freq = ignore.scale_grad_by_freq
        self.sparse = ignore.sparse

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))


    def forward(self, input):
        return F.embedding(
            input, self.weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)


    def named_leaves(self):
        return  [('weight', self.weight)]


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


# This class is modified from PyTorch source code (only supports single-directional lstm).
class LSTM(MetaModule):
    def __init__(self, *args, **kwargs):
        super(LSTM, self).__init__()
        ignore = nn.LSTM(*args, **kwargs)
        self.input_size = ignore.input_size
        self.hidden_size = ignore.hidden_size
        self.num_layers = ignore.num_layers
        self.bias = ignore.bias
        self.dropout = ignore.dropout
        self.training = ignore.training
        self.bidirectional = False
        self.batch_first = ignore.batch_first

        self._all_weights = []
        for layer in range(self.num_layers):
            suffix = ''
            weights = ['weight_ih_l{}{}', 'weight_hh_l{}{}', 'bias_ih_l{}{}', 'bias_hh_l{}{}']
            weights = [x.format(layer, suffix) for x in weights]
            if self.bias:
                self._all_weights += [weights]
            else:
                self._all_weights += [weights[:2]]

        for weights in self._all_weights:
            for weight in weights:
                self.register_buffer(weight, to_var(getattr(ignore, weight).data, requires_grad=True))


    def _flat_weights(self):
        return [getattr(self, p) for layerparams in self._all_weights for p in layerparams]


    def flatten_parameters(self):
        any_param = next(self.params()).data
        if not any_param.is_cuda or not torch.backends.cudnn.is_acceptable(any_param):
            return

        # If any parameters alias, we fall back to the slower, copying code path. This is
        # a sufficient check, because overlapping parameter buffers that don't completely
        # alias would break the assumptions of the uniqueness check in
        # Module.named_parameters().
        all_weights = self._flat_weights()
        unique_data_ptrs = set(p.data_ptr() for p in all_weights)
        if len(unique_data_ptrs) != len(all_weights):
            return

        with torch.cuda.device_of(any_param):
            import torch.backends.cudnn.rnn as rnn

            # NB: This is a temporary hack while we still don't have Tensor
            # bindings for ATen functions
            with torch.no_grad():
                # NB: this is an INPLACE function on all_weights, that's why the
                # no_grad() is necessary.
                torch._cudnn_rnn_flatten_weight(
                    all_weights, (4 if self.bias else 2),
                    self.input_size, rnn.get_cudnn_mode('LSTM'), self.hidden_size, self.num_layers,
                    self.batch_first, bool(self.bidirectional))


    def forward(self, inputs, hx):
        self.flatten_parameters()
        result = _VF.lstm(inputs, hx, self._flat_weights(), self.bias, self.num_layers,
                          self.dropout, self.training, self.bidirectional, self.batch_first)
        output = result[0]
        hidden = result[1:]

        return output, hidden


    def named_leaves(self):
        leaves = list()
        for weights in self._all_weights:
            for weight in weights:
                leaves.append((weight, getattr(self, weight)))
        return leaves


class RNNLM(MetaModule):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropoute=0.5, dropouth=0.5, dropout=0.5, batch_first=False):
        super(RNNLM, self).__init__()
        self.encoder = MetaEmbedding(vocab_size, embed_size)
        self.lstm = LSTM(embed_size, hidden_size, num_layers, batch_first=batch_first, dropout=dropouth)
        self.decoder = MetaLinear(hidden_size, vocab_size)
        self.dropoute = nn.Dropout(dropoute)
        self.dropout = nn.Dropout(dropout)

        self.init_weights()
        self.decoder.weight = self.encoder.weight


    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)


    def forward(self, x, h):
        # Embed word ids to vectors
        x = self.encoder(x)
        x = self.dropoute(x)

        # Forward propagate LSTM
        out, (h, c) = self.lstm(x, h)
        out = self.dropout(out)

        # Reshape output to (batch_size*sequence_length, hidden_size)
        out = out.reshape(out.size(0)*out.size(1), out.size(2))

        # Decode hidden states of all time steps
        out = self.decoder(out)
        return out, (h, c)
