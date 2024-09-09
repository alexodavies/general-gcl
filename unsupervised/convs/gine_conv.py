from typing import Callable, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch_sparse import SparseTensor

from unsupervised.convs.inits import reset
from inspect import Parameter

from torch_geometric.utils import (
    is_sparse,
    is_torch_sparse_tensor,
    to_edge_index,
)


from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    OrderedDict,
    Set,
    Tuple,
    Union,
)

# TODO: remove duplicates of methods (collect, propagate) that are already defined in MessagePassing
# Or don't - this works? 

class GINEConv(MessagePassing):
    def __init__(self, nn: Callable, eps: float = 0., train_eps: bool = False,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GINEConv, self).__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, edge_weight: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # Node and edge feature dimensionalites need to match.
        if isinstance(edge_index, Tensor):
            assert edge_attr is not None
            assert x[0].size(-1) == edge_attr.size(-1)
        elif isinstance(edge_index, SparseTensor):
            assert x[0].size(-1) == edge_index.size(-1)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)

        out = self.propagate(edge_index, size=size, edge_weight=edge_weight, edge_attr=edge_attr, x=x)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor, edge_attr: Tensor, edge_weight) -> Tensor:
        return F.relu(x_j + edge_attr) if edge_weight is None else F.relu(x_j + edge_attr) * edge_weight.view(-1, 1)

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)
    

    def _collect(
        self,
        args: Set[str],
        edge_index: Union[Tensor, SparseTensor],
        size: List[Optional[int]],
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:

        i, j = (1, 0) if self.flow == 'source_to_target' else (0, 1)

        out = {}
        for arg in args:
            if arg[-2:] not in ['_i', '_j']:
                out[arg] = kwargs.get(arg, Parameter.empty)
            else:
                dim = j if arg[-2:] == '_j' else i
                data = kwargs.get(arg[:-2], Parameter.empty)

                if isinstance(data, (tuple, list)):
                    assert len(data) == 2
                    if isinstance(data[1 - dim], Tensor):
                        self._set_size(size, 1 - dim, data[1 - dim])
                    data = data[dim]

                if isinstance(data, Tensor):
                    self._set_size(size, dim, data)
                    data = self._lift(data, edge_index, dim)

                out[arg] = data

        if is_torch_sparse_tensor(edge_index):
            indices, values = to_edge_index(edge_index)
            out['adj_t'] = edge_index
            out['edge_index'] = None
            out['edge_index_i'] = indices[0]
            out['edge_index_j'] = indices[1]
            out['ptr'] = None  # TODO Get `rowptr` from CSR representation.
            if out.get('edge_weight', None) is None:
                out['edge_weight'] = values
            if out.get('edge_attr', None) is None:
                out['edge_attr'] = None if values.dim() == 1 else values
            if out.get('edge_type', None) is None:
                out['edge_type'] = values

        elif isinstance(edge_index, Tensor):
            out['adj_t'] = None
            out['edge_index'] = edge_index
            out['edge_index_i'] = edge_index[i]
            out['edge_index_j'] = edge_index[j]
            out['ptr'] = None

        elif isinstance(edge_index, SparseTensor):
            row, col, value = edge_index.coo()
            rowptr, _, _ = edge_index.csr()

            out['adj_t'] = edge_index
            out['edge_index'] = None
            out['edge_index_i'] = row
            out['edge_index_j'] = col
            out['ptr'] = rowptr
            if out.get('edge_weight', None) is None:
                out['edge_weight'] = value
            if out.get('edge_attr', None) is None:
                out['edge_attr'] = value
            if out.get('edge_type', None) is None:
                out['edge_type'] = value

        out['index'] = out['edge_index_i']
        out['size'] = size
        out['size_i'] = size[i] if size[i] is not None else size[j]
        out['size_j'] = size[j] if size[j] is not None else size[i]
        out['dim_size'] = out['size_i']

        return out



    def propagate(
        self,
        edge_index: Adj,
        size: Size = None,
        **kwargs: Any,
    ) -> Tensor:
        r"""The initial call to start propagating messages.

        Args:
            edge_index (torch.Tensor or SparseTensor): A :class:`torch.Tensor`,
                a :class:`torch_sparse.SparseTensor` or a
                :class:`torch.sparse.Tensor` that defines the underlying
                graph connectivity/message passing flow.
                :obj:`edge_index` holds the indices of a general (sparse)
                assignment matrix of shape :obj:`[N, M]`.
                If :obj:`edge_index` is a :obj:`torch.Tensor`, its :obj:`dtype`
                should be :obj:`torch.long` and its shape needs to be defined
                as :obj:`[2, num_messages]` where messages from nodes in
                :obj:`edge_index[0]` are sent to nodes in :obj:`edge_index[1]`
                (in case :obj:`flow="source_to_target"`).
                If :obj:`edge_index` is a :class:`torch_sparse.SparseTensor` or
                a :class:`torch.sparse.Tensor`, its sparse indices
                :obj:`(row, col)` should relate to :obj:`row = edge_index[1]`
                and :obj:`col = edge_index[0]`.
                The major difference between both formats is that we need to
                input the *transposed* sparse adjacency matrix into
                :meth:`propagate`.
            size ((int, int), optional): The size :obj:`(N, M)` of the
                assignment matrix in case :obj:`edge_index` is a
                :class:`torch.Tensor`.
                If set to :obj:`None`, the size will be automatically inferred
                and assumed to be quadratic.
                This argument is ignored in case :obj:`edge_index` is a
                :class:`torch_sparse.SparseTensor` or
                a :class:`torch.sparse.Tensor`. (default: :obj:`None`)
            **kwargs: Any additional data which is needed to construct and
                aggregate messages, and to update node embeddings.
        """
        decomposed_layers = 1 if self.explain else self.decomposed_layers

        for hook in self._propagate_forward_pre_hooks.values():
            res = hook(self, (edge_index, size, kwargs))
            if res is not None:
                edge_index, size, kwargs = res

        mutable_size = self._check_input(edge_index, size)

        # Run "fused" message and aggregation (if applicable).
        if is_sparse(edge_index) and self.fuse and not self.explain:
            coll_dict = self._collect(self._fused_user_args, edge_index,
                                      mutable_size, kwargs)

            msg_aggr_kwargs = self.inspector.collect_param_data(
                'message_and_aggregate', coll_dict)
            for hook in self._message_and_aggregate_forward_pre_hooks.values():
                res = hook(self, (edge_index, msg_aggr_kwargs))
                if res is not None:
                    edge_index, msg_aggr_kwargs = res
            out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)
            for hook in self._message_and_aggregate_forward_hooks.values():
                res = hook(self, (edge_index, msg_aggr_kwargs), out)
                if res is not None:
                    out = res

            update_kwargs = self.inspector.collect_param_data(
                'update', coll_dict)
            out = self.update(out, **update_kwargs)

        else:  # Otherwise, run both functions in separation.
            if decomposed_layers > 1:
                user_args = self._user_args
                decomp_args = {a[:-2] for a in user_args if a[-2:] == '_j'}
                decomp_kwargs = {
                    a: kwargs[a].chunk(decomposed_layers, -1)
                    for a in decomp_args
                }
                decomp_out = []

            for i in range(decomposed_layers):
                if decomposed_layers > 1:
                    for arg in decomp_args:
                        kwargs[arg] = decomp_kwargs[arg][i]

                coll_dict = self._collect(self._user_args, edge_index,
                                          mutable_size, kwargs)

                msg_kwargs = self.inspector.collect_param_data(
                    'message', coll_dict)
                for hook in self._message_forward_pre_hooks.values():
                    res = hook(self, (msg_kwargs, ))
                    if res is not None:
                        msg_kwargs = res[0] if isinstance(res, tuple) else res
                out = self.message(**msg_kwargs)
                for hook in self._message_forward_hooks.values():
                    res = hook(self, (msg_kwargs, ), out)
                    if res is not None:
                        out = res

                if self.explain:
                    explain_msg_kwargs = self.inspector.collect_param_data(
                        'explain_message', coll_dict)
                    out = self.explain_message(out, **explain_msg_kwargs)

                aggr_kwargs = self.inspector.collect_param_data(
                    'aggregate', coll_dict)
                for hook in self._aggregate_forward_pre_hooks.values():
                    res = hook(self, (aggr_kwargs, ))
                    if res is not None:
                        aggr_kwargs = res[0] if isinstance(res, tuple) else res

                out = self.aggregate(out, **aggr_kwargs)

                for hook in self._aggregate_forward_hooks.values():
                    res = hook(self, (aggr_kwargs, ), out)
                    if res is not None:
                        out = res

                update_kwargs = self.inspector.collect_param_data(
                    'update', coll_dict)
                out = self.update(out, **update_kwargs)

                if decomposed_layers > 1:
                    decomp_out.append(out)

            if decomposed_layers > 1:
                out = torch.cat(decomp_out, dim=-1)

        for hook in self._propagate_forward_hooks.values():
            res = hook(self, (edge_index, mutable_size, kwargs), out)
            if res is not None:
                out = res

        return out