from typing import List
import html
from tqdm.auto import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from torch import mps
from sentence_transformers import SentenceTransformer, util

from InstructorEmbedding import INSTRUCTOR

class EmbeddingModel:
  def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', device=torch.device('cpu')):
    self.model_name = model_name
    self.model = SentenceTransformer(model_name).to(device)
    self.model.eval()     # using this only for inference

    self.device = device

    if device == torch.device('cuda'):
      self.empty_cache_func = torch.cuda.empty_cache
    elif device == torch.device('mps'):
      self.empty_cache_func = mps.empty_cache
    else:
      self.empty_cache_func = lambda: None

  def embed(self, messages: List[str], batch_size=128, html_unescape=True, verbose=False):
    optional_tqdm = lambda x: tqdm(x) if verbose else x

    if html_unescape:
      messages = [html.unescape(message) for message in messages]

    embeddings = []
    for i in optional_tqdm(range(0, len(messages), batch_size)):
      batch = messages[i: i+batch_size]
      with torch.no_grad():
        embeddings.extend(self.model.encode(batch, convert_to_tensor=True, device=self.device, normalize_embeddings=True).cpu().numpy())

      self.empty_cache_func()

    embeddings = np.vstack(embeddings)

    return embeddings

class InstructorEmbeddingModel:
  def __init__(self, 
               model_name='hkunlp/instructor-xl', 
               instruction='Represent the Real Estate paragraph for retrieval: ', 
               device=torch.device('cpu')):
    
    self.model_name = model_name
    self.model = INSTRUCTOR(model_name).to(device)
    self.model.eval()     # using this only for inference

    self.instruction = instruction
    self.device = device

    if device == torch.device('cuda'):
      self.empty_cache_func = torch.cuda.empty_cache
    elif device == torch.device('mps'):
      self.empty_cache_func = mps.empty_cache
    else:
      self.empty_cache_func = lambda: None

  def embed(self, messages: List[str], batch_size=32, html_unescape=True, verbose=False):
    optional_tqdm = lambda x: tqdm(x) if verbose else x

    if html_unescape:
      messages = [html.unescape(message) for message in messages]

    instructed_messages = [[self.instruction, message] for message in messages]

    embeddings = []
    for i in optional_tqdm(range(0, len(instructed_messages), batch_size)):
      with torch.no_grad():
        _embeddings = self.model.encode(instructed_messages[i: i+batch_size], convert_to_tensor=True, device=self.device, normalize_embeddings=True)
      _embeddings = _embeddings.cpu().numpy()

      embeddings.extend(_embeddings)

      self.empty_cache_func()

    embeddings = np.vstack(embeddings)

    return embeddings

