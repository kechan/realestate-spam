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

def num_tokens_from_string(string: str, encoding_name='cl100k_base') -> int:
  """Returns the number of tokens in a text string."""
  encoding = tiktoken.get_encoding(encoding_name)
  num_tokens = len(encoding.encode(string))
  return num_tokens

def truncate_string(string: str, encoding_name='cl100k_base', n_token=50):
  """ Truncates a string to n_token """
 
  encoding = tiktoken.get_encoding(encoding_name)
  tokens = encoding.encode(string)
  truncated_tokens = tokens[:n_token]
  truncated_string = encoding.decode(truncated_tokens)
  return truncated_string

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens