from typing import List, Dict, Tuple, Any

import gc
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

from .visualization import InteractiveScatter
from ..utils.misc import flatten_list

class DataDistiller:
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
  def __init__(self, df: pd.DataFrame, embedding_col: str):
    assert embedding_col in df.columns, f'{embedding_col} must be present in dataframe'
    self.df = df.copy()
    self.embedding_col = embedding_col

    self.embeddings = np.array(self.df[self.embedding_col].values.tolist())

  def generate_tsne(self, n_components=2, random_state=42, n_jobs=-1):
    tsne = TSNE(n_components=n_components, random_state=random_state, n_jobs=-1)
    tsne = tsne.fit_transform(self.embeddings)
    print(f'tsne.shape: {tsne.shape}')

    self.df['tsne'] = tsne.tolist()
    self.df.tsne = self.df.tsne.apply(np.array)

  def generate_clusters(self, n_clusters=10, random_state=42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)

    tsne = np.array(self.df.tsne.values.tolist())
    kmeans.fit(tsne)

    self.labels = kmeans.labels_
    self.centroids = kmeans.cluster_centers_


  def distill(self, n_nearest=3, n_furthest=5) -> List[Tuple[List, List]]:
    if 'tsne' not in self.df.columns:
      print('tsne column not found, generating tsne...')
      self.generate_tsne()
    
    if not hasattr(self, 'labels') or self.labels is None:  # no need to check for centroids
      print('labels not found, generating clusters...')
      self.generate_clusters()

    tsne = np.array(self.df.tsne.values.tolist())

    distilled_indices = []

    for cluster_id in range(self.centroids.shape[0]):
      centroid = self.centroids[cluster_id]

      cluster_members = tsne[self.labels == cluster_id]
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
    distilled_df = self.df.loc[flatten_list(distilled_indices)]
    print(f"dup in distill_df: {distilled_df.duplicated(subset='NOTE').sum()}")

    # the distilled set NOTE can have a lot of NOTE-duplicates because of variations in the DISPLAY_NAME. 
    # a heuristic to explore is to sample 2 from each duplicated set of NOTE. 
    
    distilled_df = distilled_df.groupby('NOTE').apply(lambda x: x.sample(n=min(2, len(x)))).reset_index(drop=True)

    return distilled_df


  def visualize_tsne(self, class_labels: List[str], color_map: Dict[str, str], with_centroids=False, width=1500, height=1500, opacity=0.2, size=4) -> Any:
    assert 'tsne' in self.df.columns, 'tsne column must be present in dataframe, please run generate_tsne() first'

    interactive_scatter = InteractiveScatter(df=self.df, class_labels=class_labels, color_map=color_map)

    fig = interactive_scatter.render_figure(width=width, height=height, opacity=opacity, size=size)

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


  

