import torch
inputs = torch.tensor(
  [ [0.43, 0.15, 0.89], # Your     (x^1)
    [0.55, 0.87, 0.66], # journey  (x^2)
    [0.57, 0.85, 0.64], # starts
    [0.22, 0.58, 0.33], # with
    [0.77, 0.25, 0.10], # one
    [0.05, 0.80, 0.55]] )# step

attn_scores = torch.empty(6, 6)

# for i, word_i in enumerate(inputs):
#     for j, word_j in enumerate(inputs):
#         attn_scores[i][j]  = torch.dot(word_i, word_j)

# Now we are calculating the attention for each word with each other word
# For loops slow, so we will use matrix multiplication instead

attn_scores = inputs @ inputs.T
attn_scores = torch.softmax(attn_scores, dim = 1)

# attn_scores for each vector with respect to every other vctor has been found

# Now to compute the context tensors
context_tensor = torch.zeros(6, 3)

for i in range(len(attn_scores[0])):
    for j, word_j in enumerate(inputs):
        context_tensor[i] = context_tensor[i] + word_j * attn_scores[i][j]
print(context_tensor)

# again, this can be done with matrix multiplication but ok, i did it with loops no worries

        














