from typing import Tuple

import pandas as pd
import numpy as np

import torch

class VectorSimSearch:
  """
    A class for performing vector similarity search on a dataframe.
    Assume embedding vector is available
  """
  def __init__(self, df: pd.DataFrame, embedding_col=None, embedding_model=None, device=torch.device('cpu')):
    self.df = df.copy()    # take a copy to ensure original df is not altered.
    assert embedding_col is not None, 'embedding_col must be provided'
    self.embedding_col = embedding_col
    self.embeddings = np.array(self.df[self.embedding_col].values.tolist())

    self.embedding_model = embedding_model
    self.device = device

  def search(self, query: str, top_n=5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a query string, return the top_n most similar.
    Returns: dataframe indices and their cosine similarity scores
    """
    assert self.embedding_model is not None, 'embedding_model must be provided'
    with torch.no_grad():
      query_embedding = self.embedding_model.encode(query, convert_to_tensor=True, device=self.device, normalize_embeddings=True)
      query_embedding = query_embedding.detach().cpu().numpy()

    cos_sim = query_embedding[None] @ self.embeddings.T

    sorted_sim = np.squeeze(np.sort(cos_sim))[::-1]
    rank = np.squeeze(np.argsort(cos_sim))[::-1]

    return rank[:top_n], sorted_sim[:top_n]
  
  def find_similar_item(self, indice: int, top_n=5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given an item's indice in the dataframe, return the top_n most similar.
    """
    try:
      query_embedding = self.df.loc[indice, self.embedding_col]      
    except:
      raise ValueError(f'indice {indice} is out of range in dataframe which has size {self.df.shape[0]}')
    
    cos_sim = query_embedding[None] @ self.embeddings.T

    sorted_sim = np.squeeze(np.sort(cos_sim))[::-1]
    rank = np.squeeze(np.argsort(cos_sim))[::-1]

    return rank[:top_n], sorted_sim[:top_n]


