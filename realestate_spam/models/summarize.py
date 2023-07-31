from typing import Any, List

import torch
from torch import mps
from tqdm.auto import tqdm

from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer

class SummarizationModel:
  @staticmethod
  def get_model_names():
    return ['t5-small', 't5-base', 't5-large', 't5-3b']

  def __init__(self, model_name='t5-large', device=torch.device('cpu')):
    self.model_name = model_name
    self.device = device

    # TODO: don't use assert on the model names, since these can be arbitrary paths (when using a local model), fix this later.
    # assert any([model_name.lower() in m.lower() for m in SummarizationModel.get_model_names()]), f'{model_name} not supported yet'

    if 't5' in model_name.lower():
      self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
      # self.tokenizer = T5Tokenizer.from_pretrained(model_name, return_dict=True)
      self.tokenizer = AutoTokenizer.from_pretrained(model_name, return_dict=True)

    # Workaround: the model name could be an arbitrary local path, if saved locally
    elif 'summarizer' in model_name.lower():
      self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
      self.tokenizer = AutoTokenizer.from_pretrained(model_name, return_dict=True)
      
    else:
      assert 'Not implemented yet, more will come soon!'

    if device == torch.device('cuda'):
      self.empty_cache_func = torch.cuda.empty_cache
    elif device == torch.device('mps'):
      self.empty_cache_func = mps.empty_cache
    else:
      self.empty_cache_func = lambda: None

  def __call__(self, *args: Any, **kwds: Any) -> Any:
    return self.summarize(*args, **kwds)

  def summarize(self, message: str, max_new_tokens=100):
    message = 'summarize: ' + message

    input_ids = self.tokenizer.encode(message, max_length=512, truncation=True, padding=True, return_tensors='pt').to(self.device)
    with torch.no_grad():
      outputs = self.model.generate(input_ids, max_new_tokens=max_new_tokens)    #, num_beams=4, early_stopping=True)

    outputs = outputs.cpu()
    self.empty_cache_func()

    return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

