import torch
import torch.nn as nn
import torch.nn.functional as F
import math


hidden_size = 2

stdv = 1.0 / math.sqrt(hidden_size)

# input_size and hidden_size = 2
W_and_U_data = torch.tensor([[stdv, stdv], [stdv, stdv]])
b_data = torch.tensor([stdv, stdv])

# z_t
W_z = nn.Parameter(W_and_U_data)
U_z = nn.Parameter(W_and_U_data)
b_z = nn.Parameter(b_data)

# r_t
W_r = nn.Parameter(W_and_U_data)
U_r = nn.Parameter(W_and_U_data)
b_r = nn.Parameter(b_data)

# h_t
W_h = nn.Parameter(W_and_U_data)
U_h = nn.Parameter(W_and_U_data)
b_h = nn.Parameter(b_data)


x = torch.tensor([[[0.5, 0.6], [0.7, 0.8]], [[0.9, 1.0], [1.1, 1.2]]])

# print(x.shape)


def forward_gru(x):
        
        """
        assumes x.shape represents (batch_size, sequence_size, input_size)
        """
        batch_size, seq_sz, _ = x.size()
        
        h_t = torch.zeros(batch_size, hidden_size)
            
        for t in range(seq_sz):
            x_t = x[:, t, :]
            z_t = torch.sigmoid(x_t @ W_z + h_t @ U_z + b_z)
            r_t = torch.sigmoid(x_t @ W_r + h_t @ U_r + b_r)
            prod_r_ht = r_t * h_t
            h_tilde = torch.tanh(x_t @ W_h + prod_r_ht * U_h + b_h)
            h_t = z_t * h_t + (1 - z_t) * h_tilde
            
        return h_t


h_t_forward = forward_gru(x)


def backward_gru(x):
        
        """
        assumes x.shape represents (batch_size, sequence_size, input_size)
        """
        batch_size, seq_sz, _ = x.size()
        
        h_t = torch.zeros(batch_size, hidden_size)
            
        for t in range(seq_sz-1, -1, -1):
            x_t = x[:, t, :]
            z_t = torch.sigmoid(x_t @ W_z + h_t @ U_z + b_z)
            r_t = torch.sigmoid(x_t @ W_r + h_t @ U_r + b_r)
            prod_r_ht = r_t * h_t
            h_tilde = torch.tanh(x_t @ W_h + prod_r_ht * U_h + b_h)
            h_t = z_t * h_t + (1 - z_t) * h_tilde
            
        return h_t


h_t_backward = backward_gru(x)


def bi_gru(h_t_forward, h_t_backward, merge_mode):
    if merge_mode == "sum":
        return h_t_forward + h_t_backward;

    if merge_mode == "mult":
        return h_t_forward * h_t_backward;

    if merge_mode == "avg":
        return (h_t_forward + h_t_backward) / 2

    if merge_mode == "concat":
        return torch.cat((h_t_forward, h_t_backward), dim=1)


merge_mode = "concat"
h_t_bi_gru = bi_gru(h_t_forward, h_t_backward, merge_mode)

print("******************************************************************\n")
print("forward gru output:\n")
print(f"{h_t_forward}\n")

print("******************************************************************\n")
print("backward gru output:\n")
print(f"{h_t_backward}\n")

print("******************************************************************\n")
print("bi gru output:\n")
print(f"{h_t_bi_gru}\n")

print("******************************************************************\n")