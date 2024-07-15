import tiktoken
import torch
import dummyGptmodel
import Gptconfig
tokenizer1 = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"
batch.append(torch.tensor(tokenizer1.encode(txt1)))
batch.append(torch.tensor(tokenizer1.encode(txt2)))
batch = torch.stack(batch, dim=0) # this stack function adds another dimension or batch to the total text

torch.manual_seed(123)
model = dummyGptmodel.DummyGPTModel(Gptconfig.GPT_CONFIG_124M)
# Logits are the name given to this output, of the backbone model
Logits = model(batch)
print(Logits.shape)
print(Logits)






    



