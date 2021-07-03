import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import itertools
import torch.nn.init as init
from torch.optim.optimizer import Optimizer

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


class MetaModule(nn.Module):
    # adopted from: Adrien Ecoffet https://github.com/AdrienLE
    def params(self):
        for name, param in self.named_params(self):
            yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:

            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)


class MetaLinear(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Linear(*args, **kwargs)

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaConv2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Conv2d(*args, **kwargs)

        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))

        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaConvTranspose2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.ConvTranspose2d(*args, **kwargs)

        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))

        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)

    def forward(self, x, output_size=None):
        output_padding = self._output_padding(x, output_size)
        return F.conv_transpose2d(x, self.weight, self.bias, self.stride, self.padding,
                                  output_padding, self.groups, self.dilation)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaBatchNorm2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super(MetaBatchNorm2d, self).__init__()
        ignore = nn.BatchNorm2d(*args, **kwargs)

        self.num_features = ignore.num_features
        self.eps = ignore.eps
        self.momentum = ignore.momentum
        self.affine = ignore.affine
        self.track_running_stats = ignore.track_running_stats

        if self.affine:
            self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(self.num_features))
            self.register_buffer('running_var', torch.ones(self.num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                            self.training or not self.track_running_stats, self.momentum, self.eps)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = metadata.get('version', None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + 'num_batches_tracked'
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super(MetaBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, metadata, strict,
            missing_keys, unexpected_keys, error_msgs)


class LSTMCell(MetaModule):

    def __init__(self, num_inputs, hidden_size):
        super(LSTMCell, self).__init__()

        self.hidden_size = hidden_size
        self.fc_i2h = MetaLinear(num_inputs, 4 * hidden_size)
        self.fc_h2h = MetaLinear(hidden_size, 4 * hidden_size)

    def init_weights(self):
        initrange = 0.1
        self.fc_h2h.weight.data.uniform_(-initrange, initrange)
        self.fc_i2h.weight.data.uniform_(-initrange, initrange)

    def forward(self, inputs, state):
        hx, cx = state
        i2h = self.fc_i2h(inputs)
        h2h = self.fc_h2h(hx)
        x = i2h + h2h
        gates = x.split(self.hidden_size, 1)

        in_gate = torch.sigmoid(gates[0])
        forget_gate = torch.sigmoid(gates[1] - 1)
        out_gate = torch.sigmoid(gates[2])
        in_transform = torch.tanh(gates[3])

        cx = forget_gate * cx + in_gate * in_transform
        hx = out_gate * torch.tanh(cx)
        return hx, cx


class MLRSNetCell(MetaModule):

    def __init__(self, num_inputs, hidden_size):
        super(MLRSNetCell, self).__init__()

        self.hidden_size = hidden_size
        self.fc_i2h = nn.Sequential(
            MetaLinear(num_inputs, hidden_size),
            nn.ReLU(),
            MetaLinear(hidden_size, 4 * hidden_size)
        )
        self.fc_h2h = nn.Sequential(
            MetaLinear(hidden_size, hidden_size),
            nn.ReLU(),
            MetaLinear(hidden_size, 4 * hidden_size)
        )
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        for module in self.fc_h2h:
            if type(module) == MetaLinear:
                module.weight.data.uniform_(-initrange, initrange)
        for module in self.fc_i2h:
            if type(module) == MetaLinear:
                module.weight.data.uniform_(-initrange, initrange)

    def forward(self, inputs, state):
        hx, cx = state
        i2h = self.fc_i2h(inputs)
        h2h = self.fc_h2h(hx)

        x = i2h + h2h
        gates = x.split(self.hidden_size, 1)

        in_gate = torch.sigmoid(gates[0])
        forget_gate = torch.sigmoid(gates[1] - 1)
        out_gate = torch.sigmoid(gates[2])
        in_transform = torch.tanh(gates[3])

        cx = forget_gate * cx + in_gate * in_transform
        hx = out_gate * torch.tanh(cx)
        return hx, cx


class MLRSNet(MetaModule):

    def __init__(self, num_layers, hidden_size):
        super(MLRSNet, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.layer1 = MLRSNetCell(1, hidden_size)
        self.layer2 = nn.Sequential(*[MLRSNetCell(hidden_size, hidden_size) for _ in range(num_layers-1)])
        self.layer3 = MetaLinear(hidden_size, 1)


    def reset_lstm(self, keep_states=False, device='cpu'):

        if keep_states:
            for i in range(len(self.layer2)+1):
                self.hx[i] = Variable(self.hx[i].data)
                self.cx[i] = Variable(self.cx[i].data)
                self.hx[i], self.cx[i] = self.hx[i].to(device), self.cx[i].to(device)
        else:
            self.hx = []
            self.cx = []
            for i in range(len(self.layer2) + 1):
                self.hx.append(Variable(torch.zeros(1, self.hidden_size)))
                self.cx.append(Variable(torch.zeros(1, self.hidden_size)))
                self.hx[i], self.cx[i] = self.hx[i].to(device), self.cx[i].to(device)


    def forward(self, x):

        if x.size(0) != self.hx[0].size(0):
            self.hx[0] = self.hx[0].expand(x.size(0), self.hx[0].size(1))
            self.cx[0] = self.cx[0].expand(x.size(0), self.cx[0].size(1))
        self.hx[0], self.cx[0] = self.layer1(x, (self.hx[0], self.cx[0]))
        x = self.hx[0]

        for i in range(1, self.num_layers):
            if x.size(0) != self.hx[i].size(0):
                self.hx[i] = self.hx[i].expand(x.size(0), self.hx[i].size(1))
                self.cx[i] = self.cx[i].expand(x.size(0), self.cx[i].size(1))

            self.hx[i], self.cx[i] = self.layer2[i-1](x, (self.hx[i], self.cx[i]))
            x = self.hx[i]

        x = self.layer3(x)
        out = torch.sigmoid(x)
        return out


class MetaSGD(Optimizer):

    def __init__(self, model, lr=0.1, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(MetaSGD, self).__init__(model.params(), defaults)
        self.model = model

    def __setstate__(self, state):
        super(MetaSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, lr, grad, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            for (name, p), g in zip(self.model.named_params(self.model), grad):
                d_p = g.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf = buf * momentum + (1 - dampening) * d_p
                    if nesterov:
                        d_p = d_p + momentum * buf
                    else:
                        d_p = buf

                tmp = p - lr * d_p
                self.model.set_param(self.model, name, tmp.squeeze(0))

        return loss
