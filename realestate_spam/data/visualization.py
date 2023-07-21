from typing import List, Dict
import plotly.graph_objs as go
import plotly.offline as pyo

import pandas as pd
import numpy as np

class InteractiveScatter:
  """
  A class for creating interactive scatter plot using plotly
  dataframe must have column named 'tsne' which contains the tsne coordinates
  """
  def __init__(self, 
               df: pd.DataFrame, 
               vector_colname='reduced_NOTE_VECTOR',
               class_label_colname='class_label', 
               class_labels: List[str]=[], 
               color_map: Dict[str, str]=[]):
    
    self.vector_colname = vector_colname
    self.class_label_colname = class_label_colname
    self.class_labels = class_labels
    self.color_map = color_map

    self.df = df.copy()
    # defrag the index
    self.df.defrag_index(inplace=True)

  def render(self, title='', width=1500, height=1500, opacity=0.2, size=4) -> go.Figure:
    """
    return plotly.graph_objs figure object
    such that you can plot this with 

    pyo.iplot(fig)  
    """

    assert self.vector_colname in self.df.columns, f'{self.vector_colname} column must be present in dataframe'
    # obtain tsne as a ndarray 
    data = np.array(self.df[self.vector_colname].values.tolist())

    self.traces = []
    for label in self.class_labels:
      indices = self.df.q(f"{self.class_label_colname} == @label").index

      # hovertext = self.df.loc[indices].NOTE.values.tolist()
      hovertext = (self.df.loc[indices].TOUCH_ENTRY_ID.astype(str) + ': ' + self.df.loc[indices].NOTE).values.tolist()

      trace = go.Scatter(x=data[indices, 0], 
                        y=data[indices, 1], 
                        mode='markers', 
                        marker=dict(
                            size=size, 
                            opacity=opacity,
                            color=self.color_map[label],  # colors,
                            # colorscale='Viridis',
                            # colorbar=dict(title='Class')
                        ),
                        hovertext=hovertext,
                        showlegend=True,
                        name=label,
                        )
      self.traces.append(trace)

    # create a layout for the plot
    self.layout = go.Layout(
      title=title,
      hovermode='closest', # show information on hover
      width=width,
      height=height,
      legend=dict(title='Class', x=0.8, y=1.0)
    )

    fig = go.Figure(data=self.traces, layout=self.layout)

    return fig
  
  def add_centroids(self, centroids: np.ndarray) -> go.Figure:
    """
    These are the centroids as computed by KMeans clustering algorithm.
    They are plotted as black dots with opacity 0.2
    """
    # Extract the centroids coordinates
    centroids_x = centroids[:, 0]
    centroids_y = centroids[:, 1]

    # Create a trace for the centroids
    centroids_trace = go.Scatter(x=centroids_x,
                                y=centroids_y,
                                mode='markers',
                                marker=dict(
                                    size=12,  # Adjust size as needed
                                    color='rgb(0, 0, 0)',  # black
                                    opacity=0.4
                                ),
                                hovertext=['Centroid {}'.format(i) for i in range(len(centroids_x))],
                                showlegend=True,
                                name='Centroids',
                                )

    # Add the centroids trace to your traces
    self.traces.append(centroids_trace)

    fig = go.Figure(data=self.traces, layout=self.layout)

    return fig

  def add_selected_points(self, points: np.ndarray) -> go.Figure:
    points_x = points[:, 0]
    points_y = points[:, 1]

    # Create a trace for the points
    points_trace = go.Scatter(x=points_x,
                              y=points_y,
                              mode='markers',
                              marker=dict(
                                  size=4,  # Adjust size as needed
                                  color='rgb(0, 0, 0)',  # black
                                  symbol = 'diamond',
                                  opacity=1.0
                              ),
                              showlegend=True,
                              name='Selected Points',
                              )
    # Add the points trace to your traces
    self.traces.append(points_trace)

    fig = go.Figure(data=self.traces, layout=self.layout)

    return fig 
                            

  