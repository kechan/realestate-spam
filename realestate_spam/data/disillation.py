from typing import List, Dict, Tuple, Any

from abc import ABC, abstractmethod

import gc
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

from .visualization import InteractiveScatter
from ..utils.misc import flatten_list

class DataDistiller(ABC):
  """
  Distill dataset into a smaller, representative set. 
  It is best to start from a dataset that has been "simply" de-duplicated.

  General approach:
  1. dimenstionality reduction (e.g. tSNE)
  2. clustering (e.g. KMeans)
  3. grid sampling

  A strategy can involve a combination of the above steps.
  """

  def __init__(self, df: pd.DataFrame, embedding_col: str):
    assert embedding_col in df.columns, f'{embedding_col} must be present in dataframe'

    self.df = df.defrag_index().copy()   # defrag is important for data alignment
    self.embedding_col = embedding_col

    self.embeddings = np.array(self.df[self.embedding_col].values.tolist())

  @abstractmethod
  def dim_reduce(self, dim=2, **kwargs):
    raise NotImplementedError
  
  @abstractmethod
  def generate_clusters(self, n_clusters=10, **kwargs):
    raise NotImplementedError
  
  @abstractmethod
  def distill(self, **kwargs) -> pd.DataFrame:
    # final algorithm to distill and return the distilled dataset
    raise NotImplementedError


class KMeansSamplingDataDistiller(DataDistiller):
  """
  Distill dataset into a much smaller, representative dataset. Useful for:
  - Quick Testing, faster training
  - Few shot training set for in context learning for LLMs

  Approach:
  1. use tSNE to obtain 2d projection of the databset embeddings
  2. use KMeans clustering to cluster the embeddings
  3. Extract nearest N and furthest M members from each cluster
  4. return the distilled dataset

  """
  def dim_reduce(self, dim=2, random_state=42, n_jobs=-1):
    reduced_dim_embeddings = self._generate_tsne(n_components=dim, random_state=random_state, n_jobs=n_jobs)

    # store the reduced dim embeddings in the dataframe
    self.df[f'reduced_{self.embedding_col}'] = reduced_dim_embeddings.tolist()
    self.df[f'reduced_{self.embedding_col}'] = self.df[f'reduced_{self.embedding_col}'].apply(np.array)


  def _generate_tsne(self, n_components=2, random_state=42, n_jobs=-1):
    tsne = TSNE(n_components=n_components, random_state=random_state, n_jobs=-1)
    reduced_dim_embeddings = tsne.fit_transform(self.embeddings)
    print(f'tsne.shape: {reduced_dim_embeddings.shape}')
    return reduced_dim_embeddings


  def generate_clusters(self, n_clusters=10, random_state=42):
    """
    Generate clusters using KMeans
    store the labels and centroids in self.labels and self.centroids (labels are membership indices)
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)

    reduced_dim_embeddings = np.array(self.df[f'reduced_{self.embedding_col}'].values.tolist())
    kmeans.fit(reduced_dim_embeddings)

    self.labels = kmeans.labels_
    self.centroids = kmeans.cluster_centers_


  def distill(self, n_nearest=3, n_furthest=5, random_state=42) -> List[Tuple[List, List]]:
    if f'reduced_{self.embedding_col}' not in self.df.columns:
      print(f'reduce_{self.embedding_col} column not found, generating them first...')
      self.dim_reduce(random_state=random_state)
    
    assert not (not hasattr(self, 'labels') or self.labels is None), print('cluster labels not found, run .generate_clusters(...) first')

    reduced_embeddings = np.array(self.df[f'reduce_{self.embedding_col}'].values.tolist())

    distilled_indices = []

    for cluster_id in range(self.centroids.shape[0]):
      centroid = self.centroids[cluster_id]

      cluster_members = reduced_embeddings[self.labels == cluster_id]
      # print(f'# of members in cluster {cluster_id}: {cluster_members.shape[0]}')
      # distances = np.sqrt(np.sum((cluster_members - centroid)**2, axis=1))
      distances = np.linalg.norm(cluster_members - centroid, axis=1)

      sorted_indices = np.argsort(distances)   # indice r.p.t. to cluster_members

      # indices of members beloging to this cluster 
      member_idx = np.where(self.labels == cluster_id)[0]

      nearest_nn_idx = member_idx[sorted_indices[: n_nearest]].tolist()
      furthest_nn_idx = member_idx[sorted_indices[-n_furthest:]].tolist()

      distilled_indices.append([nearest_nn_idx, furthest_nn_idx])

    # flatten the list and obtain the selected rows from self.df  
    print(flatten_list(distilled_indices))
    distilled_df = self.df.loc[flatten_list(distilled_indices)]
    print(f"dup in distill_df: {distilled_df.duplicated(subset='NOTE').sum()}")

    # the distilled set NOTE can have a lot of NOTE-duplicates because of variations in the DISPLAY_NAME. 
    # a heuristic to explore is to sample 2 from each duplicated set of NOTE. 
    
    distilled_df = distilled_df.groupby('NOTE').apply(lambda x: x.sample(n=min(2, len(x)))).reset_index(drop=True)

    return distilled_df


  def visualize(self, class_labels: List[str], color_map: Dict[str, str], with_centroids=False, width=1500, height=1500, opacity=0.2, size=4) -> Any:
    assert f'reduced_{self.embedding_col}' in self.df.columns, 'f"reduced_{self.embedding_col}" column must be present in dataframe, please run .dim_reduce() first'

    interactive_scatter = InteractiveScatter(df=self.df, vector_colname=f'reduce_{self.embedding_col}', class_labels=class_labels, color_map=color_map)

    fig = interactive_scatter.render(width=width, height=height, opacity=opacity, size=size)

    if with_centroids and hasattr(self, 'centroids'):
      fig = interactive_scatter.add_centroids(self.centroids)
    
    return fig

  def reset(self):
    # drop tsne column from self.df
    self.df.drop(columns=['tsne'], inplace=True)

    # reset labels and centroids
    if hasattr(self, 'labels'): self.labels = None
    if hasattr(self, 'centroids'): self.centroids = None

    gc.collect()


class UniformGridSamplingDataDistiller(DataDistiller):
  """
  Approach:
  1. use tSNE to obtain 2d projection of the databset embeddings
  2. divide the 2d space into N x N grid and uniformly sample M points from each cell

  """
  def dim_reduce(self, dim=2, random_state=42, n_jobs=-1):
    reduced_dim_embeddings = self._generate_tsne(n_components=dim, random_state=random_state, n_jobs=n_jobs)

    # store the reduced dim embeddings in the dataframe
    self.df[f'reduced_{self.embedding_col}'] = reduced_dim_embeddings.tolist()
    self.df[f'reduced_{self.embedding_col}'] = self.df[f'reduced_{self.embedding_col}'].apply(np.array)

    # reference the reduced dim embeddings in self.reduce_dim_embeddings for further processing 
    self.reduce_dim_embeddings = reduced_dim_embeddings


  def _generate_tsne(self, n_components=2, random_state=42, n_jobs=-1):
    tsne = TSNE(n_components=n_components, random_state=random_state, n_jobs=-1)
    reduced_dim_embeddings = tsne.fit_transform(self.embeddings)
    print(f'tsne.shape: {reduced_dim_embeddings.shape}')
    return reduced_dim_embeddings
  
  def generate_clusters(self, n_clusters=10, **kwargs):
    raise NotImplementedError
  
  def distill(self, N: int, M: int) -> pd.DataFrame:
    """
    Sampling method:
    1. divide the 2d space into N x N grid
    2. uniformly sample M points from each grid

    """
    e = self.reduce_dim_embeddings    # for simpler code
    
    x_min, x_max = e[:, 0].min(), e[:, 0].max()
    y_min, y_max = e[:, 1].min(), e[:, 1].max()
    # x_grid = np.linspace(x_min, x_max, N)
    # y_grid = np.linspace(y_min, y_max, N)

    # Compute the size of each cell
    x_step = (x_max - x_min) / N
    y_step = (y_max - y_min) / N

    # Initialize the grid as a list of empty lists
    grid = [[[] for _ in range(N)] for _ in range(N)]
    population_number_grid = np.zeros((N, N), dtype=np.int8)

    # Populate the grid, e.g. grid[i][j] is a list of data points whose coordinates are in the cell (i, j)
    for i in range(e.shape[0]):
      x, y = e[i]
      grid_index_x = min(int((x - x_min) / x_step), N - 1)
      grid_index_y = min(int((y - y_min) / y_step), N - 1)
      grid[grid_index_x][grid_index_y].append(i)

      population_number_grid[grid_index_x][grid_index_y] += 1

    self.population_number_grid = population_number_grid   # store it for later use 

    # Sample M points from each cell
    samples = []
    for i in range(N):
      for j in range(N):
        if len(grid[i][j]) > 0:
          cell_samples = np.random.choice(grid[i][j], min(M, len(grid[i][j])), replace=False)
          samples.extend(cell_samples)

    distilled_df = self.df.loc[samples].copy()
    distilled_df.defrag_index(inplace=True)

    return distilled_df






