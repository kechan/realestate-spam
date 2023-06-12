from typing import List
import html
from tqdm.auto import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from torch import mps
from sentence_transformers import SentenceTransformer, util

class EmbeddingModel:
  def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', device=torch.device('cpu')):
    self.model_name = model_name
    self.model = SentenceTransformer(model_name).to(device)
    self.model.eval()     # using this only for inference

    if device == torch.device('cuda'):
      self.empty_cache_func = torch.cuda.empty_cache
    elif device == torch.device('mps'):
      self.empty_cache_func = mps.empty_cache
    else:
      self.empty_cache_func = lambda: None

  def embed(self, messages: List[str], batch_size=128, html_unescape=True):
    if html_unescape:
      messages = [html.unescape(message) for message in messages]

    embeddings = []
    for i in tqdm(range(0, len(messages), batch_size)):
      batch = messages[i: i+batch_size]
      with torch.no_grad():
        embeddings.extend(self.model.encode(batch, convert_to_tensor=True).cpu().numpy())

      self.empty_cache_func()

    embeddings = np.vstack(embeddings)

    return embeddings

   