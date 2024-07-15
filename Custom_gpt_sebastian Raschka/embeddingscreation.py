import torch    
import dataloaderbatchsizing



vocab_size = 50257
output_dim = 256   
# Here we are making a embeddings neural network using the torch library nn, this will create 
# an absolute embedding with no relation to its position and will create here, a 256 dimension word embedding
torch.manual_seed(123)
embeddings_layer = torch.nn.Embedding(vocab_size, output_dim)
dataiter = iter(dataloaderbatchsizing.dataloader1)
input, targets = next(dataiter)
token_embeddings = embeddings_layer(input)
# print(token_embeddings.shape)

# As we know the position of the word will also play a role in the meaning, to understand context
# thus another embedding is required to show the position that the word occupies

pos_embedding_layer = torch.nn.Embedding(dataloaderbatchsizing.context_size, 256) # the vocab is limited to 4 as the window length is 4
pos_embedding = pos_embedding_layer(torch.arange(dataloaderbatchsizing.context_size))
# print(pos_embedding.shape)

# Now to make the final embedding that we will give to the transformer
final_embedding = token_embeddings + pos_embedding
# Its a bit scary as we think it will not add dimensionally, but python is smart and will understand out intentions
# print(final_embedding.shape) # matches the expected output






