from typing import List, Any
from tqdm.auto import tqdm

import pandas as pd
import torch
import torch.nn.functional as F
from torch import mps

from transformers import AutoModelForSequenceClassification, AutoTokenizer

import html, gc

class ToxicDetectionModel:
  def __init__(self, model_name='jpcorb20/toxic-detector-distilroberta', device=torch.device('cpu')):
    self.model_name = model_name
    self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    self.model.eval()     # using this only for inference

    self.device = device

    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self._class_names = self.model.config.id2label
    self.class_names = [v for _, v in self._class_names.items()]

    if device == torch.device('cuda'):
      self.empty_cache_func = torch.cuda.empty_cache
    elif device == torch.device('mps'):
      self.empty_cache_func = mps.empty_cache
    else:
      self.empty_cache_func = lambda: None

  def __call__(self, *args: Any, **kwds: Any) -> Any:
    return self.get_toxicity(*args, **kwds)

  def get_toxicity(self, messages: List[str], batch_size=128, html_unescape=True, return_df=True, verbose=False) -> pd.DataFrame:
    """Returns a list of toxicity scores for each message in messages.
    
    Args:
      messages: A list of strings.
      html_escape: If True, html escapes each message before passing it to the model.
    
    Returns:
      A pandas dataframe holding the results
    """
    optional_tqdm = lambda x: tqdm(x) if verbose else x

    if isinstance(messages, str): messages = [messages]
    
    probs = {'message': []}
    for class_name in self.class_names: probs[class_name] = []

    if html_unescape:
      messages = [html.unescape(message) for message in messages]

    # for message in messages:
    for i in optional_tqdm(range(0, len(messages), batch_size)):
      batch = messages[i: i+batch_size]

      probs['message'].extend(batch)

      # Tokenize the messages, and convert to tensors
      # input_ids = self.tokenizer.encode(message, return_tensors='pt', truncation=True)
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
      toxic_df = pd.DataFrame(probs)
      return toxic_df
    else:
      return probs
      


    
    
