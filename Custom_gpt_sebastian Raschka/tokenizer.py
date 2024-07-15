import re

def clean_rtf(text, x):

    text = re.sub(r'\\cb[13]', '', text)
    text = re.sub(r'\\[a-z]+\d*', '', text)
    text = re.sub(r'[{}]', '', text)
    text = re.sub(r'\\', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if x > 0:
        text = text[x:]

    return text
with open('the-verdict.rtf', 'r') as file:
    text = file.read()

x = 22
cleaned_text = clean_rtf(text, x)


# Scheme one simple split based in space bar
# token1 = re.split( r'(\s)', cleaned_text)
# num = 0
# for i in token1:
#     num = num + 1
#     print(i)
#     if num == 100 :
#         break

# Scheme 2 based on commas and fullstops with previous space bar and all types of punctuation marks
token2 = re.split(r'([,.:;?_!"()\']|--|\s)', cleaned_text)
token2 = [item.strip() for item in token2 if item.strip()]
num2 = 0

# Next we try to build a dictionary or a vocabulary of the words
all_words = sorted(list(set(token2)))
all_words.extend(["<|endoftext|>", "<|unk|>"])
vocab_length = len(all_words)
# print(all_words[-5:])


vocab = {word : i for i, word in enumerate(all_words)}  # the syntax for making the vocab out of the list of words
# for word, key in vocab.items():
#     if key < 10 :
#         print(word)
    





