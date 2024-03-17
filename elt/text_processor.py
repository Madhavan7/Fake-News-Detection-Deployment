import re
from nltk.stem.porter import PorterStemmer
from transformers import AutoTokenizer


class TextProcessor():
  def __init__(self, max_length):
    self.max_length = max_length
    self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
  
  def tokenize(self, seq):
    #returns the attention mask
    tokens = self.tokenizer(seq, return_tensors="np", padding='max_length', max_length=self.max_length, truncation=True)
    return tokens['input_ids'][0].tolist()

  def mask(self, seq):
    tokens = self.tokenizer(seq, return_tensors="np", padding='max_length', max_length=self.max_length, truncation=True)
    return tokens['attention_mask'][0].tolist()
  
#simple test to make sure dimensions check out
# processor = TextProcessor(512)
# tokens = processor.tokenize("Hello.   MY    name is @Madhav, and I am coding. I am 23 years old")
# mask = processor.mask("Hello.   MY    name is @Madhav, and I am coding. I am 23 years old")

# print(tokens)
# print(mask)