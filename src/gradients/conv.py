from src.gradients.grad import Function, unbroadcast
import numpy as np
from numba import njit, prange

@njit(parallel=True)
def _conv1d_forward_kernel(data, weight, out, stride, kernel_size):
    batch_size, in_channels, in_length = data.shape
    out_channels = weight.shape[0]
    out_length = out.shape[2]
    stride_val = stride[0]
    
    for i in prange(batch_size):
        for j in range(out_channels):
            for k in range(out_length):
                l_start = k * stride_val
                val = 0.0
                for m in range(in_channels):
                    for n in range(kernel_size):
                        val += data[i, m, l_start + n] * weight[j, m, n]
                out[i, j, k] = val

@njit(parallel=True)
def _conv1d_backward_kernel(data, weight, grad, stride, kernel_size):
    batch_size, in_channels, in_length = data.shape
    out_channels = weight.shape[0]
    out_length = grad.shape[2]
    stride_val = stride[0]
    
    grad_node = np.zeros((batch_size, in_channels, in_length), dtype=data.dtype)
    grad_weight = np.zeros((batch_size, out_channels, in_channels, kernel_size), dtype=data.dtype)
    
    for i in prange(batch_size):
        for j in range(out_channels):
            for k in range(out_length):
                l_start = k * stride_val
                g = grad[i, j, k]
                for m in range(in_channels):
                    for n in range(kernel_size):
                        grad_node[i, m, l_start + n] += g * weight[j, m, n]
        
        for j in range(out_channels):
            for m in range(in_channels):
                for n in range(kernel_size):
                    val = 0.0
                    for k in range(out_length):
                        val += grad[i, j, k] * data[i, m, k * stride_val + n]
                    grad_weight[i, j, m, n] = val
                    
    return grad_node, grad_weight

@njit(parallel=True)
def _conv2d_forward_kernel(data, weight, out, stride, kernel_height, kernel_width):
    batch_size, in_channels, in_height, in_width = data.shape
    out_channels = weight.shape[0]
    out_height, out_width = out.shape[2:]
    stride_h, stride_w = stride[0], stride[1]
    
    for i in prange(batch_size):
        for j in range(out_channels):
            for k in range(out_height):
                h_start = k * stride_h
                for l in range(out_width):
                    w_start = l * stride_w
                    val = 0.0
                    for m in range(in_channels):
                        for n in range(kernel_height):
                            for p in range(kernel_width):
                                val += data[i, m, h_start + n, w_start + p] * weight[j, m, n, p]
                    out[i, j, k, l] = val

@njit(parallel=True)
def _conv2d_backward_kernel(data, weight, grad, stride, kernel_height, kernel_width):
    batch_size, in_channels, in_height, in_width = data.shape
    out_channels = weight.shape[0]
    out_height, out_width = grad.shape[2:]
    stride_h, stride_w = stride[0], stride[1]
    
    grad_node = np.zeros((batch_size, in_channels, in_height, in_width), dtype=data.dtype)
    grad_weight = np.zeros((batch_size, out_channels, in_channels, kernel_height, kernel_width), dtype=data.dtype)
    
    for i in prange(batch_size):
        for j in range(out_channels):
            for k in range(out_height):
                h_start = k * stride_h
                for l in range(out_width):
                    w_start = l * stride_w
                    g = grad[i, j, k, l]
                    for m in range(in_channels):
                        for n in range(kernel_height):
                            for p in range(kernel_width):
                                grad_node[i, m, h_start + n, w_start + p] += g * weight[j, m, n, p]
        
        for j in range(out_channels):
            for m in range(in_channels):
                for n in range(kernel_height):
                    for p in range(kernel_width):
                        val = 0.0
                        for k in range(out_height):
                            for l in range(out_width):
                                val += grad[i, j, k, l] * data[i, m, k * stride_h + n, l * stride_w + p]
                        grad_weight[i, j, m, n, p] = val
                        
    return grad_node, grad_weight

class Convolution(Function):
    @staticmethod
    def forward(ctx, node, weight, stride, padding, bias):
        # weight.shape -> out_channels, in_channels, ... (nd)
        ctx.saved_for_backward(node, weight, bias)
        ctx.stride = stride
        ctx.padding = padding

        kernel_size = weight.shape[2:]

        if len(node.shape) == len(weight.shape)-1 : 
            out_shape = [weight.shape[0]] + [int((node.shape[1 + i]+2*p-k)/s)+1 for i, (k, s, p) in enumerate(zip(kernel_size, stride, padding))]
            pad_width = [(0, 0)] +  [(p, p) for p in padding]
        elif len(node.shape) == len(weight.shape):
            out_shape = [node.shape[0], weight.shape[0]] + [int((node.shape[2+i]+2*p-k)/s)+1 for i, (k, s, p) in enumerate(zip(kernel_size, stride, padding))]
            pad_width = [(0, 0), (0, 0)] +  [(p, p) for p in padding]
        else :
            raise ValueError(f"Inputs shape should be {len(weight.shape)-1}D(Channels, ...) or {len(weight.shape)}D(Batch_size, Channels, ..)")
            
        node_data = np.pad(node.data, pad_width=pad_width)

        if len(weight.shape) == 3: # Conv1d
            ret = Conv1d_forward(node_data, weight.data, out_shape, stride)
        elif len(weight.shape) == 4: # Conv2d
            ret = Conv2d_forward(node_data, weight.data, out_shape, stride)
        else:
            raise NotImplementedError(f"Conv{len(weight.shape)-2}d operation is not implemented")

        if bias is not None:
            ret += bias.data
        return ret
    
    @staticmethod
    def backward(ctx, grad=1):
        # grad shape: out_shape((b) c, l)
        node, weight, bias = ctx.saved_tensors    
        stride = ctx.stride
        padding = ctx.padding

        if len(node.shape) == len(weight.shape) -1:
            batch_first = False
            pad_width = [(0, 0)] +  [(p, p) for p in padding]
        elif len(node.shape) == len(weight.shape):
            batch_first = True
            pad_width = [(0, 0), (0, 0)] +  [(p, p) for p in padding]
        
        node_data = np.pad(node.data, pad_width=pad_width)

        if len(weight.shape) == 3: # Conv1d
            grad_node, grad_weight = Conv1d_backward(node_data, weight.data, grad, stride, padding, batch_first)
        elif len(weight.shape) == 4: # Conv2d
            grad_node, grad_weight = Conv2d_backward(node_data, weight.data, grad, stride, padding, batch_first)
        else:
            raise NotImplementedError(f"Conv{len(weight.shape)-2}d operation is not implemented")

        grad_node = unbroadcast(grad_node, node.shape)
        grad_weight = unbroadcast(grad_weight, weight.shape)
        grad_bias = unbroadcast(grad, bias.shape)
        return grad_node, grad_weight, grad_bias if bias is not None else None

def Conv1d_forward(data, weight, out_shape, stride):
    if data.ndim == 2:
        data_3d = data[np.newaxis, ...]
        out_3d = np.zeros((1, *out_shape), dtype=data.dtype)
    else:
        data_3d = data
        out_3d = np.zeros(out_shape, dtype=data.dtype)
    
    kernel_size = weight.shape[2]
    _conv1d_forward_kernel(data_3d, weight, out_3d, np.array(stride), kernel_size)
    
    if data.ndim == 2:
        return out_3d[0]
    return out_3d

def Conv1d_backward(data, weight, grad, stride, padding, batch_first):
    if not batch_first:
        data_3d = data[np.newaxis, ...]
        grad_3d = grad[np.newaxis, ...]
    else:
        data_3d = data
        grad_3d = grad
        
    kernel_size = weight.shape[2]
    grad_node_3d, grad_weight_4d = _conv1d_backward_kernel(data_3d, weight, grad_3d, np.array(stride), kernel_size)
    
    in_length = data_3d.shape[2]
    grad_node_3d = grad_node_3d[..., padding[0]:in_length-padding[0]]
    
    if not batch_first:
        return grad_node_3d[0], grad_weight_4d[0]
    return grad_node_3d, grad_weight_4d

def Conv2d_forward(data, weight, out_shape, stride):
    if data.ndim == 3:
        data_4d = data[np.newaxis, ...]
        out_4d = np.zeros((1, *out_shape), dtype=data.dtype)
    else:
        data_4d = data
        out_4d = np.zeros(out_shape, dtype=data.dtype)
    
    kernel_height, kernel_width = weight.shape[2:]
    _conv2d_forward_kernel(data_4d, weight, out_4d, np.array(stride), kernel_height, kernel_width)
    
    if data.ndim == 3:
        return out_4d[0]
    return out_4d

def Conv2d_backward(data, weight, grad, stride, padding, batch_first):
    if not batch_first:
        data_4d = data[np.newaxis, ...]
        grad_4d = grad[np.newaxis, ...]
    else:
        data_4d = data
        grad_4d = grad
        
    kernel_height, kernel_width = weight.shape[2:]
    grad_node_4d, grad_weight_5d = _conv2d_backward_kernel(data_4d, weight, grad_4d, np.array(stride), kernel_height, kernel_width)
    
    in_height, in_width = data_4d.shape[-2:]
    grad_node_4d = grad_node_4d[..., padding[0]:in_height-padding[0], padding[1]:in_width-padding[1]]
    
    if not batch_first:
        return grad_node_4d[0], grad_weight_5d[0]
    return grad_node_4d, grad_weight_5d
