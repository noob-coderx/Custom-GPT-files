import torch
import torch.nn as nn
import DeepNeuralNetwork as DNN # my own classes, not some preexisting python module, refer from the book
import Gptconfig as cfg

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = DNN.MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"], 
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = DNN.FeedForward(cfg)
        self.norm1 = DNN.LayerNorm(cfg["emb_dim"])
        self.norm2 = DNN.LayerNorm(cfg["emb_dim"])
        self.drop_resid = nn.Dropout(cfg["drop_rate"])
 
    def forward(self, x):
        #A
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_resid(x)
        x = x + shortcut  # Add the original input back
 
        shortcut = x #B
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_resid(x)
        x = x + shortcut  #C 
        return x
                  

