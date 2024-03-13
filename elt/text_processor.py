import re
from nltk.stem.porter import PorterStemmer
from transformers import AutoTokenizer


MAX_LENGTH = 512

class TextProcessor():
  def __init__(self):
    self.stemmer = PorterStemmer()
    self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
  
  def lemmatize(self, text):
    text = text.lower()
    re.sub(r'\W+', ' ', text)
    t_array = [self.stemmer.stem(t) for t in text.split()]
    text = " ".join(t_array)
    return text
  
  def tokenize(self, text):
    #returns the attention mask
    text = self.lemmatize(text)
    tokens = self.tokenizer(text, return_tensors="pt", padding='max_length', max_length=MAX_LENGTH, truncation=True)
    return tokens['input_ids'], tokens['attention_mask']
  
#simple test to make sure dimensions check out
processor = TextProcessor()
tokens, mask = processor.tokenize("Hello.   MY    name is @Madhav, and I am coding. I am 23 years old")
assert mask.shape[0] == 1 and mask.shape[1] == 512
assert tokens.shape[0] == 1 and tokens.shape[1] == 512

print(tokens)
