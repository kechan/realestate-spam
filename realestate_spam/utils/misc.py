from typing import List, Tuple

from itertools import chain
import html, tiktoken

def show_overlap(df1, df2, label=None, col=None, display_cols=[]):
  assert col is not None, 'col must be provided'
  if label is None:
    overlaps = set(df1[col].values).intersection(set(df2[col].values))
  else:
    overlaps = set(df1.q(f"class_label == '{label}'")[col].values).intersection(set(df2.q(f"class_label == '{label}'")[col].values))

  if None in overlaps:
    overlaps.remove(None)

  if label is None:
    return overlaps, df1.q(f"{col}.isin(@overlaps)")[display_cols], df2.q(f"{col}.isin(@overlaps)")[display_cols]
  else:
    return overlaps, df1.q(f"class_label == '{label}' and {col}.isin(@overlaps)")[display_cols], df2.q(f"class_label == '{label}' and {col}.isin(@overlaps)")[display_cols]

def html_unescape(df):
  df.NOTE = df.NOTE.apply(html.unescape)

def flatten_list(tuples_list: List[Tuple[List, List]]) -> List:
  flattened_list = list(chain.from_iterable(chain(*tuples_list)))
  return flattened_list

def num_tokens_from_string(string: str, encoding_name: str) -> int:
  """Returns the number of tokens in a text string."""
  encoding = tiktoken.get_encoding(encoding_name)
  num_tokens = len(encoding.encode(string))
  return num_tokens