import torch
inputs = torch.tensor(
  [ [0.43, 0.15, 0.89], # Your     (x^1)
    [0.55, 0.87, 0.66], # journey  (x^2)
    [0.57, 0.85, 0.64], # starts
    [0.22, 0.58, 0.33], # with
    [0.77, 0.25, 0.10], # one
    [0.05, 0.80, 0.55]] )# step


# To find the attention weight we will do the dot product and find the weights, that we refer to as omega or alpha
query = inputs[1]  #A
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)

# Normalisation
# attn_scores_2 = attn_scores_2/attn_scores_2.sum()

# Normalisation using softmax, to be precise torch's softmax
attn_scores_2 = torch.softmax(attn_scores_2, dim=0)
print(attn_scores_2)

# Now lets multiply the vectors to find the context vector
context_vec_2 = torch.zeros(query.shape)
for i,x_i in enumerate(inputs):
    context_vec_2 += attn_scores_2[i]*x_i
print(context_vec_2)