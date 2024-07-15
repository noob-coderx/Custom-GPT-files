import torch
import GPT  
import tiktoken
def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimensi
    return encoded_tensor
def generate_text_simple(model, idx, max_new_tokens, context_size): #A
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:] #B
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :] #C
        probas = torch.softmax(logits, dim=-1)  #D
        idx_next = torch.argmax(probas, dim=-1, keepdim=True) #E
        idx = torch.cat((idx, idx_next), dim=1)  #F
    return idx
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,  #A
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,     #B
    "qkv_bias": False
}

torch.manual_seed(123)
model = GPT.GPTModel(GPT_CONFIG_124M)
start_context = "Every effort moves you"
tokenizer1 = tiktoken.get_encoding("gpt2")
token_ids = generate_text_simple(model=model,idx=text_to_token_ids(start_context, tokenizer1),max_new_tokens=10,context_size=GPT_CONFIG_124M["context_length"])
inputs = torch.tensor([[16833, 3626, 6100],   # ["every effort moves",
                       [40,    1107, 588]])   #  "I really like"]
targets = torch.tensor([[3626, 6100, 345  ],  # [" effort moves you",
                        [588,  428,  11311]]) #  " really like chocolate"]
with torch.no_grad(): #A
    logits = model(inputs)
# print(logits.shape)
# print(targets.shape)
logits_flat = logits.flatten(0, 1)
targets_flat = targets.flatten()
# This is bascially merging the batches into a large batch, dosent matter to have them seperated anymore as
# the output from the transformers has already arrived

#Cross-entropy loss is basically the -log loss 
loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
print(loss)
# This loss is a useful measure for optimization of models






