GPT_CONFIG_124M = {
"vocab_size": 50257,  # Vocabulary size
"context_length": 512,      # Context length  256 for the timebeing will be set to 1024 after this when we load parameter values from the gpt2 model
"emb_dim": 768, # Embedding dimensions
"n_heads": 12, # number of attention heads
"n_layers": 12, #Number of layers
"drop_rate": 0.1, #Dropout rate in the attention weights
"qkv_bias": False #query-key value bias, for the nn linear module 
}