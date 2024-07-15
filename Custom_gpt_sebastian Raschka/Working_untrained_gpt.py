import torch
import tiktoken
import GPT
import Gptconfig as cfg



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

start_context = "Hello, I am"
tokenizer1 = tiktoken.get_encoding("gpt2")
encoded = tokenizer1.encode(start_context)
print("encoded", encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print(encoded_tensor.shape)
model = GPT.GPT
model.eval()
out = generate_text_simple(model, encoded_tensor, 6, cfg.GPT_CONFIG_124M["context_length"])
print(out)
print(len(out[0]))

decoded_text = tokenizer1.decode(out.squeeze(0).tolist())
print(decoded_text)
# the model is untrained and gives random stuff as output




