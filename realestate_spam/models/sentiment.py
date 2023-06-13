from typing import List

from transformers import AutoModelForSequenceClassification, AutoTokenizer

import html, gc
import pandas as pd
import torch
from torch import mps

from tqdm.auto import tqdm

class SentimentClassificationModel:
  def __init__(self, model_name='siebert/sentiment-roberta-large-english', device=torch.device('cpu')):
    self.model_name = model_name
    self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    self.model.eval()   # using this only for inference

    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self._class_names = self.model.config.id2label
    self.class_names = [v for _, v in self._class_names.items()]

    if device == torch.device('cuda'):
      self.empty_cache_func = torch.cuda.empty_cache
    elif device == torch.device('mps'):
      self.empty_cache_func = mps.empty_cache
    else:
      self.empty_cache_func = lambda: None

  def get_sentiment(self, messages: List[str], batch_size=64, html_unescape=True, return_df=True, verbose=False) -> pd.DataFrame:
    optional_tqdm = lambda x, verbose=False: tqdm(x) if verbose else x

    if isinstance(messages, str): messages = [messages]

    probs = {'message': [], 'NEGATIVE': [], 'POSITIVE': []}
    for i in optional_tqdm(range(0, len(messages), batch_size)):
      
      batch = messages[i: i+batch_size]
      probs['message'].extend(batch)
      if html_unescape:
        batch = [html.unescape(message) for message in batch]
      
      inputs = self.tokenizer(batch, return_tensors='pt', max_length=512, truncation=True, padding=True)
      for k, v in inputs.items(): inputs[k] = v.to(self.model.device)

      with torch.no_grad():
        outputs = self.model(**inputs)

      probabilities = torch.sigmoid(outputs.logits)
      probabilities = probabilities.detach().cpu().numpy()

      for class_idx, p in enumerate(probabilities.T):
        class_name = self._class_names[class_idx]
        probs[class_name].extend(p)

      gc.collect()
      self.empty_cache_func()

    if return_df:
      return pd.DataFrame(probs)
    else:
      return probs
    



