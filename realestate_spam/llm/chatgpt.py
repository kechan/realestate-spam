from typing import List, Dict
from raa import RAA    

import json

from ..utils.misc import num_tokens_from_string

class FirstTouchNoteSpamDetectorAgent:
  def __init__(self, 
               system_prompt: str = None, 
               guidelines: List[str] = None, 
               spam_n_shot_samples: List[str] = None,
               not_spam_n_shot_samples: List[str] = None,
               test_n_shot_samples: List[str] = None):
    
    self.system_prompt = system_prompt
    self.guidelines = guidelines
    self.spam_n_shot_samples = spam_n_shot_samples
    self.not_spam_n_shot_samples = not_spam_n_shot_samples
    self.test_n_shot_samples = test_n_shot_samples

    self.inputs = None    # placeholder for prediction inputs

    assert self.spam_n_shot_samples is not None, 'spam_n_shot_samples must be provided'
    assert self.not_spam_n_shot_samples is not None, 'not_spam_n_shot_samples must be provided'
    assert self.test_n_shot_samples is not None, 'test_n_shot_samples must be provided'

    if self.guidelines is None:   # default available
      self.guidelines = [
        "If the message is related to buying, selling, renting, or inquiring about a property, then it is NOT_SPAM.",
        "If the message concerns home valuation, real estate career inquiry, etc., it is NOT_SPAM",
        "Personal narrative, story, or anecdote that is unrelated to real estate, classify it as SPAM",
        "If the sender is promoting services such as SEO, web site traffic optimization, photography, personal brand promotions, painting services, etc., classify them as SPAM.",
        "If names look random string, bias it towards being a SPAM. However, if the message is related to (1), classify it as NOT_SPAM.",
        "Messages that look like testing during development or support, classify them as TEST.",
        "If a message contains phrases or terms such as \"This is a Test\", \"Test this\", \"Testing\", \"Please Ignore\", or any variant indicating that the message is for development or testing purposes, classify them as TEST, even if they contain other content related to real estate.",
        "If a message contains explicit or inappropriate content, regardless of the display name or intent, classify it as SPAM.",
        "If a message is not in English, try not to be biased. But if it is in Russian, just be extra cautious.",
        "If you are not sure, classify as \"NOT_SURE\"",
      ]
    guidelines_as_str = '\n'.join([f"{i+1}) {g}" for i, g in enumerate(self.guidelines)])

    # format n_shot_samples as long string
    n_shot_sample_for_spam = '\n'.join([f'SPAM, {s}' for s in spam_n_shot_samples])
    print(f'# token used for spam samples: {num_tokens_from_string(n_shot_sample_for_spam)}')
    n_shot_sample_for_not_spam = '\n'.join([f'NOT_SPAM, {s}' for s in not_spam_n_shot_samples])
    print(f'# token used for not_spam samples: {num_tokens_from_string(n_shot_sample_for_not_spam)}')
    n_shot_sample_for_test = '\n'.join([f'TEST, {s}' for s in test_n_shot_samples])
    print(f'# token used for test samples: {num_tokens_from_string(n_shot_sample_for_test)}')

    if system_prompt is None:   # use default 
      self.system_prompt = f"""You are a spam classifier able to classify messages into NOT_SPAM, SPAM, and TEST for a real estate agent app. 
The body of each message will be preceded by "DISPLAY_NAME is ... " to specify the display name of the sender. 

Please use the following guidelines: 
{guidelines_as_str}

Here are a few examples for you to learn:
{n_shot_sample_for_spam}
{n_shot_sample_for_not_spam}
{n_shot_sample_for_test}
"""

    self.raa = RAA(sys_prompt=self.system_prompt)    # chatgpt completion API wrapped up as an agent

  def predict(self, texts: List[str]) -> List[Dict]: 

    inputs = '\n'.join([f"<input>{text}</input>" for text in texts])
    user_prompt = f"""
    Can you please classify the following messages delimited by <inputs> and </inputs>? 
<inputs>
{inputs}
</inputs>
please provide predictions for each with either "NOT_SPAM", "SPAM", "TEST" or "NOT_SURE" together with your reasonings in less than 20 words 
in the format of json that is a list of dict with keys "Prediction" and "Reasoning". Provide only the JSON output without any additional text or explanation.
"""
    messages = [{"role": "user", "content": user_prompt}]
    predictions_str = self.raa.get_completion_from_messages(messages, temperature=0.1, max_tokens=500, debug=False)

    # expected return is a json formatted answer
    # verify if predictions_str parsed to a dict (i.e. valid json)
    try:
      predictions = json.loads(predictions_str)
    except:
      print(f'Warning: malformed json: {predictions_str}')
      print(f'trying to extract the json part heuristically...')
      predictions_str = self._extract_valid_json(predictions_str)

      # try parse json again
      try:
        predictions = json.load(predictions_str)
      except:
        raise ValueError(f'Unrecoverable malformed json: {predictions_str}')
    
    return predictions


  def _extract_valid_json(self, s):
    """
    Extract the largest valid JSON substring from s.

    Once in a while, GPT may not return a well-formed json, and this is a heuristic attempt
    to recover

    NB: One shouldn't expect this too often, so double for loops is fine, otherwise
    this code may be slow (as a brute force approach). For longer string, you can try
    to call _fast_extract_valid_json and see how it does, before calling this one.
    """
    for i in range(len(s)):
      for j in range(len(s), i, -1):
        try:
          data = json.loads(s[i:j])
          return json.dumps(data)  # Return a valid JSON string.
        except json.JSONDecodeError:
          pass
    return None
  
  def _fast_extract_valid_json(self, s):
    """
    Extract valid JSON substring from s.

    This 
    """
    opening_chars = ['{', '[']
    closing_chars = ['}', ']']

    for i, char in enumerate(s):
      if char in opening_chars:
        balance = 1
        corresponding_closing = closing_chars[opening_chars.index(char)]
        for j in range(i + 1, len(s)):
          if s[j] == char:
            balance += 1
          elif s[j] == corresponding_closing:
            balance -= 1

            if balance == 0:  # Found a matching closing char
              try:
                data = json.loads(s[i:j+1])
                return json.dumps(data)
              except json.JSONDecodeError:
                break  # Move on to the next opening character
    return None

    

    