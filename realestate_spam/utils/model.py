from typing import Any
import pandas as pd
from tqdm import tqdm
import tensorflow as tf

def run_model_evaluation(df: pd.DataFrame, tokenizer: Any, model: Any, batch_size=1024):
  '''
  Convenience method to evaluate model 

  params:
    df: pd.DataFrame
    tokenizer: transformers.tokenizer
    model: transformers.model
    batch_size: int
  '''
  assert 'TOUCH_ENTRY_ID' in df.columns, 'df must have "TOUCH_ENTRY_ID" column'
  assert 'text' in df.columns and 'label' in df.columns, 'df must have "text" and "label" columns'

  touch_entry_ids, notes, predictionss, labelss, probss = [], [], [], [], []

  for k in tqdm(range(0, df.shape[0], batch_size)):
    texts = df.iloc[k: k+batch_size].text.values.tolist()
    inputs = tokenizer(texts, truncation=True, padding=True, return_tensors='tf')
    outputs = model(inputs).logits

    probs = tf.nn.softmax(outputs, axis=1).numpy()
    predictions = tf.argmax(outputs, axis=1).numpy()
    labels = df.iloc[k: k+batch_size].label.values

    for i in range(len(texts)):
      notes.append(df.iloc[k+i].NOTE)
      predictionss.append(predictions[i])
      labelss.append(labels[i])
      probss.append(probs[i])
      touch_entry_ids.append(df.iloc[k+i].TOUCH_ENTRY_ID)

  result_df = pd.DataFrame(data={
    'TOUCH_ENTRY_ID': touch_entry_ids, 
    'note': notes, 
    'prediction': predictionss, 
    'label': labelss, 
    'p': probss}
    )

  return result_df


def run_model_inference(df: pd.DataFrame, tokenizer: Any, model: Any, batch_size=1024):
  assert 'TOUCH_ENTRY_ID' in df.columns, 'df must have "TOUCH_ENTRY_ID" column'
  assert 'text' in df.columns, 'df must have "text" column'

  touch_entry_ids, notes, predictionss, probss = [], [], [], []

  for k in tqdm(range(0, df.shape[0], batch_size)):
    texts = df.iloc[k: k+batch_size].text.values.tolist()
    inputs = tokenizer(texts, truncation=True, padding=True, return_tensors='tf')
    outputs = model(inputs).logits

    probs = tf.nn.softmax(outputs, axis=1).numpy()
    predictions = tf.argmax(outputs, axis=1).numpy()

    for i in range(len(texts)):
      notes.append(df.iloc[k+i].NOTE)
      predictionss.append(predictions[i])
      probss.append(probs[i])
      touch_entry_ids.append(df.iloc[k+i].TOUCH_ENTRY_ID)

  result_df = pd.DataFrame(data={
    'TOUCH_ENTRY_ID': touch_entry_ids, 
    'note': notes, 
    'prediction': predictionss, 
    'p': probss}
    )

  return result_df