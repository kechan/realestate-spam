from typing import List, Any
from tqdm.auto import tqdm

import html, gc
import pandas as pd
import torch
import torch.nn.functional as F
from torch import mps

from transformers import AutoModelForSequenceClassification, AutoTokenizer

class ZeroShotClassificationModel:
  def __init__(self, model_name='facebook/bart-large-mnli', class_names: List[str]=None, device=torch.device('cpu')):
    """
    Args:
      model_name: The name of the model to use. See https://huggingface.co/models for a list of models.
      class_names: A list of class names for the zero-shot classification task
      device: The device to use for inference. Defaults to cpu.
    """
    assert class_names is not None and isinstance(class_names, list), 'class_names must be provided as a List[str]'
    self.class_names = class_names
    self.hypotheses = [f'This example is about {label}.' for label in class_names]

    self.model_name = model_name
    self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    self.model.eval()     # using this only for inference

    self.device = device

    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
 
    if device == torch.device('cuda'):
      self.empty_cache_func = torch.cuda.empty_cache
    elif device == torch.device('mps'):
      self.empty_cache_func = mps.empty_cache
    else:
      self.empty_cache_func = lambda: None

  def __call__(self, messages: List[str], batch_size=128, html_unescape=True, return_df=True, aggressive_clear_cache=False, verbose=False) -> pd.DataFrame:
    optional_tqdm = lambda x: tqdm(x) if verbose else x

    if isinstance(messages, str): messages = [messages]

    # probs as dict to hold the prediction score for each class
    probs = {'message': []}
    probs = {**probs, **{class_name: [] for class_name in self.class_names}}

    for i in optional_tqdm(range(0, len(messages), batch_size)):
      batch = messages[i: i+batch_size]
      probs['message'].extend(batch)

      if html_unescape:
        batch = [html.unescape(message) for message in batch]

      for hypothesis, class_name in zip(self.hypotheses, self.class_names):
        inputs = self.tokenizer(batch, [hypothesis]*len(batch), return_tensors='pt', truncation='only_first', padding=True)
        for k, v in inputs.items(): inputs[k] = v.to(self.device)

        with torch.no_grad():
          outputs = self.model(**inputs)
          logits = outputs.logits
          prob = logits[:, [0, 2]].softmax(dim=1)[:,1].cpu().numpy()

        probs[class_name].extend(prob.tolist())
        if aggressive_clear_cache:
          gc.collect()
          self.empty_cache_func()

      gc.collect()
      self.empty_cache_func()

    if return_df:
      return pd.DataFrame(probs)
    else:
      return probs


