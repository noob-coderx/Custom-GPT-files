import tiktoken
import tokenizer
import torch
from torch.utils.data import DataLoader, Dataset
tokenizer3 = tiktoken.get_encoding("gpt2")
integers = tokenizer3.encode(tokenizer.cleaned_text, allowed_special={"<|endoftext|>"})
# print(integers)

# Awesome BPE works with our text 

enc_sample = integers[50:]
def Createdataloader(txt, batch_size=4,max_length=256, stride=128, shuffle=True, drop_last=True):
    toke = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, toke, max_length, stride)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last = drop_last)
    return dataloader
# x is the inputs and y is the output

context_size = 4
x = enc_sample[0:context_size]
y = enc_sample[1:context_size + 1]
for i in range(1, context_size + 1):
    context =  enc_sample[0:i]
    desired = enc_sample[i]
    # here the context is the data that will be fed to the model and desired will be the prediction that the model is expected to make
    #obviously

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt) #A
        for i in range(0, len(token_ids) - max_length, stride): #B
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    def __len__(self): #C
        return len(self.input_ids)
    def __getitem__(self, idx): #D
        return self.input_ids[idx], self.target_ids[idx]

dataloader1 = Createdataloader(tokenizer.cleaned_text, 8, 4, 4, False)
# dataiter = iter(dataloader1)
# first_batch_inputs, first_batch_targets = next(dataiter)
# print(first_batch_inputs)
# print("-------")
# print(first_batch_targets)









