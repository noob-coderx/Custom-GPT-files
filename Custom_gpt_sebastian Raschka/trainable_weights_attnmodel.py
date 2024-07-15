import torch
inputs = torch.tensor(
  [ [0.43, 0.15, 0.89], # Your     (x^1)
    [0.55, 0.87, 0.66], # journey  (x^2)
    [0.57, 0.85, 0.64], # starts
    [0.22, 0.58, 0.33], # with
    [0.77, 0.25, 0.10], # one
    [0.05, 0.80, 0.55]] )# step

x_2 = inputs[1]
d_in = inputs.shape[1] # dimension of encoding
d_out = 2

torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value

keys = inputs @ W_key
values = inputs @ W_value

attn_scores = query_2 @ keys.T
# this is will do the dot product between the input 
d_k = keys.shape[-1]   # for better training the attn_scores are divided by the length of each key
attn_weights = torch.softmax(attn_scores/d_k**0.5, dim = 0)

#Here we finally scale and add the value vectors to find out the context vector

context_vec_2 = attn_weights @ values
print(context_vec_2)

