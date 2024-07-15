import torch
import torch.nn as nn
import Transformer
import DeepNeuralNetwork as DNN
import Gptconfig as cfg
import tiktoken

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        self.trf_blocks = nn.Sequential(*[Transformer.TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
       
        self.final_norm = DNN.LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )
 
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        #A
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

torch.manual_seed(123)
tokenizer1 = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"
batch.append(torch.tensor(tokenizer1.encode(txt1)))
batch.append(torch.tensor(tokenizer1.encode(txt2)))
batch = torch.stack(batch, dim=0) # this stack function adds another dimension or batch to the total text
GPT = GPTModel(cfg.GPT_CONFIG_124M)
logits = GPT.forward(batch)
# print(logits)
# print(logits.shape)
total_parameters = sum(p.numel() for p in GPT.parameters())
# print(total_parameters) # WOAH thats a crazy amount
# print("Token embedding layer shape:", GPT.tok_emb.weight.shape)
# print("Output layer shape:", GPT.out_head.weight.shape)
# print(trf_parameters = sum(p.numel() for p in GPT.trf_blocks.parameters()))
# Look up what numel is exactly








