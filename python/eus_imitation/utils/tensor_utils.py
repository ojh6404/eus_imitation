#!/usr/bin/env python3

import collections
from typing import List, Tuple, Dict, Union, Any, Callable, Optional
import numpy as np
import torch



# def recursive_apply(x , type_fn_dict):
def recursive_apply(
    x: Union[Dict, List, Tuple, torch.Tensor, np.ndarray],
    type_fn_dict: Dict[type, Callable],
) -> Union[Dict, List, Tuple, torch.Tensor, np.ndarray]:
    assert list not in type_fn_dict
    assert tuple not in type_fn_dict
    assert dict not in type_fn_dict

    if isinstance(x, (dict, collections.OrderedDict)):
        return type(x)([(k, recursive_apply(v, type_fn_dict)) for k, v in x.items()])
    elif isinstance(x, (list, tuple)):
        return type(x)([recursive_apply(v, type_fn_dict) for v in x])
    else:
        for type_, fn in type_fn_dict.items():
            if isinstance(x, type_):
                return fn(x)
        else:
            raise NotImplementedError("No type matched for {}".format(x))


def map_tensor(x, fn):
    return recursive_apply(x, {torch.Tensor: fn, type(None): lambda x: x})


def map_ndarray(x, fn):
    return recursive_apply(x, {np.ndarray: fn, type(None): lambda x: x})


def map_tensor_ndarray(x, tensor_fn, ndarray_fn):
    return recursive_apply(
        x, {torch.Tensor: tensor_fn, np.ndarray: ndarray_fn, type(None): lambda x: x}
    )


def clone(x):
    return recursive_apply(
        x,
        {
            torch.Tensor: lambda x: x.clone(),
            np.ndarray: lambda x: x.copy(),
            type(None): lambda x: x,
        },
    )


def detach(x):
    return recursive_apply(
        x,
        {
            torch.Tensor: lambda x: x.detach(),
            np.ndarray: lambda x: x,
            type(None): lambda x: x,
        },
    )


def to_batch(x):
    return recursive_apply(
        x,
        {
            torch.Tensor: lambda x: x[None, ...],
            np.ndarray: lambda x: x[None, ...],
            type(None): lambda x: x,
        },
    )


def to_sequence(x):
    return recursive_apply(
        x,
        {
            torch.Tensor: lambda x: x[:, None, ...],
            np.ndarray: lambda x: x[:, None, ...],
            type(None): lambda x: x,
        },
    )


def index_at_time(x, index):
    return recursive_apply(
        x,
        {
            torch.Tensor: lambda x: x[:, index, ...],
            np.ndarray: lambda x: x[:, index, ...],
            type(None): lambda x: x,
        },
    )

def squeeze(x, dim):
    return recursive_apply(
        x,
        {
            torch.Tensor: lambda x: x.squeeze(dim=dim),
            np.ndarray: lambda x: np.squeeze(x, axis=dim),
            type(None): lambda x: x,
        },
    )

def unsqueeze(x, dim):
    return recursive_apply(
        x,
        {
            torch.Tensor: lambda x: x.unsqueeze(dim=dim),
            np.ndarray: lambda x: np.expand_dims(x, axis=dim),
            type(None): lambda x: x,
        },
    )


def contiguous(x):
    return recursive_apply(
        x,
        {
            torch.Tensor: lambda x: x.contiguous(),
            np.ndarray: lambda x: np.ascontiguousarray(x),
            type(None): lambda x: x,
        },
    )


def to_device(x, device):
    return recursive_apply(
        x,
        {
            torch.Tensor: lambda x: x.to(device),
            np.ndarray: lambda x: x,
            type(None): lambda x: x,
        },
    )


def to_tensor(x):
    return recursive_apply(
        x,
        {
            torch.Tensor: lambda x: x,
            np.ndarray: lambda x: torch.from_numpy(x),
            float: lambda x: torch.tensor(x),
            type(None): lambda x: x,
        },
    )


def to_numpy(x):
    def device_to_cpu(x):
        if x.is_cuda:
            return x.detach().cpu()
        else:
            return x.detach()

    return recursive_apply(
        x,
        {
            torch.Tensor: lambda x: device_to_cpu(x).numpy(),
            np.ndarray: lambda x: x,
            type(None): lambda x: x,
        },
    )


def to_list(x):
    def device_to_cpu(x):
        if x.is_cuda:
            return x.detach().cpu()
        else:
            return x.detach()

    return recursive_apply(
        x,
        {
            torch.Tensor: lambda x: device_to_cpu(x).tolist(),
            np.ndarray: lambda x: x.tolist(),
            type(None): lambda x: x,
        },
    )


def to_float(x):
    return recursive_apply(
        x,
        {
            torch.Tensor: lambda x: x.float(),
            np.ndarray: lambda x: x.astype(np.float32),
            type(None): lambda x: x,
        },
    )


def to_uint8(x):
    return recursive_apply(
        x,
        {
            torch.Tensor: lambda x: x.uint8(),
            np.ndarray: lambda x: x.astype(np.uint8),
            type(None): lambda x: x,
        },
    )


def to_torch(x, device):
    return to_device(to_float(to_tensor(x)), device)


def to_one_hot_single(tensor, num_class):
    x = torch.zeros(tensor.size() + (num_class,)).to(tensor.device)
    x.scatter_(-1, tensor.unsqueeze(-1), 1)
    return x


def to_one_hot(tensor, num_class):
    return map_tensor(tensor, fn=lambda x: to_one_hot_single(x, num_class))


def flatten_single(x, begin_axis=1):
    fixed_size = x.size()[:begin_axis]
    _s = list(fixed_size) + [-1]
    return x.reshape(*_s)


def flatten(x, begin_axis=1):
    return recursive_apply(x, {torch.Tensor: lambda x: flatten_single(x, begin_axis)})


def reshape_dimensions_single(x, begin_axis, end_axis, target_dims):
    assert begin_axis <= end_axis
    assert begin_axis >= 0
    assert end_axis < len(x.shape)
    assert isinstance(target_dims, (tuple, list))
    s = x.shape
    final_s = []
    for i in range(len(s)):
        if i == begin_axis:
            final_s.extend(target_dims)
        elif i < begin_axis or i > end_axis:
            final_s.append(s[i])
    return x.reshape(*final_s)


def reshape_dimensions(x, begin_axis, end_axis, target_dims):
    return recursive_apply(
        x,
        {
            torch.Tensor: lambda x: reshape_dimensions_single(
                x, begin_axis, end_axis, target_dims
            ),
            np.ndarray: lambda x: reshape_dimensions_single(
                x, begin_axis, end_axis, target_dims
            ),
            type(None): lambda x: x,
        },
    )


def join_dimensions(x, begin_axis, end_axis):
    return recursive_apply(
        x,
        {
            torch.Tensor: lambda x: reshape_dimensions_single(
                x, begin_axis, end_axis, target_dims=[-1]
            ),
            np.ndarray: lambda x: reshape_dimensions_single(
                x, begin_axis, end_axis, target_dims=[-1]
            ),
            type(None): lambda x: x,
        },
    )


def expand_at_single(x, size, dim):
    assert dim < x.ndimension()
    assert x.shape[dim] == 1
    expand_dims = [-1] * x.ndimension()
    expand_dims[dim] = size
    return x.expand(*expand_dims)


def expand_at(x, size, dim):
    return map_tensor(x, fn=lambda x: expand_at_single(x, size, dim))


def unsqueeze_expand_at(x, size, dim):
    return expand_at(unsqueeze(x, dim), size, dim)


def repeat_by_expand_at(x, repeats, dim):
    return join_dimensions(unsqueeze_expand_at(x, repeats, dim + 1), dim, dim + 1)


def pad_sequence_single(seq, padding, batched=False, pad_same=True, pad_values=None):
    assert isinstance(seq, (np.ndarray, torch.Tensor))
    assert pad_same or pad_values is not None
    if pad_values is not None:
        assert isinstance(pad_values, float)
    repeat_func = np.repeat if isinstance(seq, np.ndarray) else torch.repeat_interleave
    concat_func = np.concatenate if isinstance(seq, np.ndarray) else torch.cat
    ones_like_func = np.ones_like if isinstance(seq, np.ndarray) else torch.ones_like
    seq_dim = 1 if batched else 0

    begin_pad = []
    end_pad = []

    if padding[0] > 0:
        pad = seq[[0]] if pad_same else ones_like_func(seq[[0]]) * pad_values
        begin_pad.append(repeat_func(pad, padding[0], seq_dim))
    if padding[1] > 0:
        pad = seq[[-1]] if pad_same else ones_like_func(seq[[-1]]) * pad_values
        end_pad.append(repeat_func(pad, padding[1], seq_dim))

    return concat_func(begin_pad + [seq] + end_pad, seq_dim)


def pad_sequence(seq, padding, batched=False, pad_same=True, pad_values=None):
    return recursive_apply(
        seq,
        {
            torch.Tensor: lambda x: pad_sequence_single(
                x, padding, batched, pad_same, pad_values
            ),
            np.ndarray: lambda x: pad_sequence_single(
                x, padding, batched, pad_same, pad_values
            ),
            type(None): lambda x: x,
        },
    )


def assert_size_at_dim_single(x, size, dim, msg):
    assert x.shape[dim] == size, msg


def assert_size_at_dim(x, size, dim, msg):
    map_tensor(x, lambda t, s=size, d=dim, m=msg: assert_size_at_dim_single(t, s, d, m))


def get_shape(x):
    return recursive_apply(
        x,
        {
            torch.Tensor: lambda x: x.shape,
            np.ndarray: lambda x: x.shape,
            type(None): lambda x: x,
        },
    )


def list_of_flat_dict_to_dict_of_list(list_of_dict):
    assert isinstance(list_of_dict, list)
    dic = collections.OrderedDict()
    for i in range(len(list_of_dict)):
        for k in list_of_dict[i]:
            if k not in dic:
                dic[k] = []
            dic[k].append(list_of_dict[i][k])
    return dic


def flatten_nested(d, parent_key="", sep="_", item_key=""):
    items = []
    if isinstance(d, (tuple, list)):
        new_key = parent_key + sep + item_key if len(parent_key) > 0 else item_key
        for i, v in enumerate(d):
            items.extend(flatten_nested(v, new_key, sep=sep, item_key=str(i)))
        return items
    elif isinstance(d, dict):
        new_key = parent_key + sep + item_key if len(parent_key) > 0 else item_key
        for k, v in d.items():
            assert isinstance(k, str)
            items.extend(flatten_nested(v, new_key, sep=sep, item_key=k))
        return items
    else:
        new_key = parent_key + sep + item_key if len(parent_key) > 0 else item_key
        return [(new_key, d)]


def time_distributed(
    inputs, op, activation=None, inputs_as_kwargs=False, inputs_as_args=False, **kwargs
):
    batch_size, seq_len = flatten_nested(inputs)[0][1].shape[:2]
    inputs = join_dimensions(inputs, 0, 1)
    if inputs_as_kwargs:
        outputs = op(**inputs, **kwargs)
    elif inputs_as_args:
        outputs = op(*inputs, **kwargs)
    else:
        outputs = op(inputs, **kwargs)

    if activation is not None:
        outputs = map_tensor(outputs, activation)
    outputs = reshape_dimensions(
        outputs, begin_axis=0, end_axis=0, target_dims=(batch_size, seq_len)
    )
    return outputs


if __name__ == "__main__":
    tensorA = torch.randn(2, 3, 4, 5, 6)

    print(tensorA.shape)

    reshaped = reshape_dimensions_single(tensorA, 0, 3, target_dims=[1])

    print(reshaped.shape)
