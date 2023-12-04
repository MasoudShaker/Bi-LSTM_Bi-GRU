import torch
import torch.nn as nn
import torch.nn.functional as F
import math


hidden_size = 2

stdv = 1.0 / math.sqrt(hidden_size)

# input_size and hidden_size = 2
W_and_U_data = torch.tensor([[stdv, stdv], [stdv, stdv]])
b_data = torch.tensor([stdv, stdv])

#i_t
W_i = nn.Parameter(W_and_U_data)
U_i = nn.Parameter(W_and_U_data)
b_i = nn.Parameter(b_data)

#f_t
W_f = nn.Parameter(W_and_U_data)
U_f = nn.Parameter(W_and_U_data)
b_f = nn.Parameter(b_data)

#c_t
W_c = nn.Parameter(W_and_U_data)
U_c = nn.Parameter(W_and_U_data)
b_c = nn.Parameter(b_data)

#o_t
W_o = nn.Parameter(W_and_U_data)
U_o = nn.Parameter(W_and_U_data)
b_o = nn.Parameter(b_data)


x = torch.tensor([[[0.5, 0.6], [0.7, 0.8]], [[0.9, 1.0], [1.1, 1.2]]])

# print(x.shape)

# x_t = x[:, 1, :]

# print(x_t.shape)

def forward_lstm(x):
        
        """
        assumes x.shape represents (batch_size, sequence_size, input_size)
        """
        batch_size, seq_sz, _ = x.size()
        
        h_t, c_t = torch.zeros(batch_size, hidden_size), torch.zeros(batch_size, hidden_size)
            
        for t in range(seq_sz):
            x_t = x[:, t, :]
            
            i_t = torch.sigmoid(x_t @ W_i + h_t @ U_i + b_i)
            f_t = torch.sigmoid(x_t @ W_f + h_t @ U_f + b_f)
            g_t = torch.tanh(x_t @ W_c + h_t @ U_c + b_c)
            o_t = torch.sigmoid(x_t @ W_o + h_t @ U_o + b_o)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            
        return h_t


h_t_forward = forward_lstm(x)


def backward_lstm(x):
        
        """
        assumes x.shape represents (batch_size, sequence_size, input_size)
        """
        batch_size, seq_sz, _ = x.size()
        
        h_t, c_t = torch.zeros(batch_size, hidden_size), torch.zeros(batch_size, hidden_size)
            
        for t in range(seq_sz-1, -1, -1):
            x_t = x[:, t, :]
            
            i_t = torch.sigmoid(x_t @ W_i + h_t @ U_i + b_i)
            f_t = torch.sigmoid(x_t @ W_f + h_t @ U_f + b_f)
            g_t = torch.tanh(x_t @ W_c + h_t @ U_c + b_c)
            o_t = torch.sigmoid(x_t @ W_o + h_t @ U_o + b_o)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            
        return h_t


h_t_backward = backward_lstm(x)


def bi_lstm(h_t_forward, h_t_backward, merge_mode):
    if merge_mode == "sum":
        return h_t_forward + h_t_backward;

    if merge_mode == "mult":
        return h_t_forward * h_t_backward;

    if merge_mode == "avg":
        return (h_t_forward + h_t_backward) / 2

    if merge_mode == "concat":
        return torch.cat((h_t_forward, h_t_backward), dim=1)
    

merge_mode = "concat"
h_t_bi_lstm = bi_lstm(h_t_forward, h_t_backward, merge_mode)


print("******************************************************************\n")
print("forward lstm output:\n")
print(f"{h_t_forward}\n")


print("******************************************************************\n")
print("backward lstm output:\n")
print(f"{h_t_backward}\n")


print("******************************************************************\n")
print("bi lstm output:\n")
print(f"{h_t_bi_lstm}\n")

print("******************************************************************")