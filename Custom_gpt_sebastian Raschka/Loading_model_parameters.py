import torch
from GPT import GPTModel
from Gptconfig import GPT_CONFIG_124M

model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict("model.pth")
optimizer = torch.optim.AdamW(model.parameters(), lr = 5e-4, weight_decay = 0.1)
optimizer.load_state_dict("optimizer_state_dict.pth")
model.eval()

# Somw issue is present in the loading phase of the optimizer, that is causing issue
#Am moving on for now to the other parts will look into this later
