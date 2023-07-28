from typing import List


import langdetect
from langdetect import detect_langs

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel

import torch
from torch import mps

import pandas as pd
from tqdm.auto import tqdm

try:
  from cld2 import detect as cld2_detect
except:
  print('unable to import cld2')

try:
  import cld3
  print(f'import cld3 successful')
except:
  print('unable to import cld3')

try:
  import gcld3
  print(f'import gcld3 successful')
except:
  print('unable to import gcld3')

class EfficientLanguageDetector:
  """
    The idea is to use lighter weight lang detect first, if it fails or low probability, before 
    trying to use 'papluca/xlm-roberta-base-language-detection' which is more accurate but resource intensive
  """
  
  def __init__(self, 
               model_name='papluca/xlm-roberta-base-language-detection', 
               try_langdetect_only=False, 
               use_gcld3=True, 
               device=torch.device('cpu')):
    self.model_name = model_name
    self.try_langdetect_only = try_langdetect_only
    self.use_gcld3 = use_gcld3
    self.device = device

    self.lang_detect_model = None  # lazy load this
    if use_gcld3:
      self.gcld3_identifier = None # lazy load this

    try:
      cld2_detect('dummy')
      self.mixed_encoding_checking = True
    except:
      self.mixed_encoding_checking = False    # Was unable to import cld2, skipped checking for mixed encoding

    print('EfficientLanguageDetector initiated')

  def detect(self, message, verbose=False):
    # check if message is empty or just spaces
    if message is None or len(message.strip()) == 0:
      return {'message': message, 'lang': None, 'prob': 0.0}
    
    result = self.light_detect(message)

    if result['prob'] < 0.95:
      # lazy load the bigger heavier model
      if self.lang_detect_model is None: self.lang_detect_model = LanguageDetector(model_name=self.model_name, device=self.device)

      result = self.lang_detect_model.detect([message], batch_size=1, verbose=verbose).iloc[0]
      result = {'message': result['message'], 'lang': result['lang'], 'prob': float(result['prob'])}

    return result


  def light_detect(self, message: str):
    if self.try_langdetect_only:
      return self._try_lang_detect(message)
    else:
      try:
        if self.use_gcld3:
          if self.gcld3_identifier is None: self.gcld3_identifier = gcld3.NNetLanguageIdentifier(min_num_bytes=0, max_num_bytes=1024)
          result = self.gcld3_identifier.FindLanguage(message)
        else:
          result = cld3.get_language(message)
        # a hack, use cld2 to trigger an exception in order to detect mixed/wrong encoding
        # cld3 is susceptible to wrong prediction in windows related encoded as utf-8
        if self.mixed_encoding_checking:
          cld2_detect(message)

        if result is not None and result.is_reliable:
          return {'message': message, 'lang': result.language, 'prob': result.probability}
        else:
          result = self._try_lang_detect(message)

      except (TypeError, ValueError, UnicodeDecodeError) as e:
        result = self._try_lang_detect(message)
    
    if isinstance(result, list) and len(result) > 0 and isinstance(result[0], langdetect.language.Language):
      return {'message': message, 'lang': result[0].lang, 'prob': result[0].prob}
    else:
      return {'message': message, 'lang': None, 'prob': 0.0}
    
  
  def _try_lang_detect(self, message: str):
    # langdetect is not very accurate on short random text, but it is fast in general
    # print('trying again with langdetect')
    try:
      result = detect_langs(message)
    except Exception as e:
      # print(f'lang detect failed: {e}')
      return {'message': message, 'lang': None, 'prob': 0.0}
    
    return {'message': message, 'lang': result[0].lang, 'prob': result[0].prob}



class LanguageDetector:
  def __init__(self, model_name='papluca/xlm-roberta-base-language-detection', device=torch.device('cpu')):
    self.model_name = model_name
    self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.model.eval()   # using this only for inference

    self.device = device

    if device == torch.device('cuda'):
      self.empty_cache_func = torch.cuda.empty_cache
    elif device == torch.device('mps'):
      self.empty_cache_func = mps.empty_cache
    else:
      self.empty_cache_func = lambda: None

  def detect(self, messages: List[str], batch_size=128, verbose=False):
    optional_tqdm = lambda x, verbose=False: tqdm(x) if verbose else x

    langs, probs = [], []
    for i in optional_tqdm(range(0, len(messages), batch_size), verbose=verbose):
      inputs = self.tokenizer(messages[i: i+batch_size], return_tensors='pt', truncation=True, padding=True)
      for k, v in inputs.items(): inputs[k] = v.to(self.device)

      with torch.no_grad():
        outputs = self.model(**inputs)

      probs += torch.max(torch.softmax(outputs.logits, dim=1), dim=1).values.cpu().tolist()
      langs += [self.model.config.id2label[i] for i in torch.argmax(outputs.logits, dim=1).cpu().numpy()]

      self.empty_cache_func()

    return pd.DataFrame({'message': messages, 'lang': langs, 'prob': probs})

