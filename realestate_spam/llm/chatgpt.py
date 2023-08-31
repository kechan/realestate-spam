from typing import List, Dict
from raa import RAA    

import json

from ..utils.misc import num_tokens_from_string

class DisplayNameOriginAgent:
  def __init__(self, system_prompt: str = None, 
              user_prompt_turn_1: str = None, 
              ai_response_turn_1: str = None):
    self.system_prompt = system_prompt
    self.user_prompt_turn_1 = user_prompt_turn_1
    self.ai_response_turn_1 = ai_response_turn_1

    if self.system_prompt is None:
      self.system_prompt = """
You are an expert on people first and last name and their origin. Below is a few examples. 
Note that there will be "display_name" that are not real, and you should answer with "false" for "is_name". 
Please ensure that any special characters, especially quotation marks, within the JSON values are properly escaped 
using a backslash (e.g., \"). This will ensure the output remains valid JSON format and is parsable. 
Provide only the JSON output without any additional text or explanation.
"""
    if self.user_prompt_turn_1 is None:
      self.user_prompt_turn_1 = "Max Williams, Ram Kumar, dfgdf KfnqDuxw"
    
    if self.ai_response_turn_1 is None:
      self.ai_response_turn_1 = """
[{"name": "Max Williams", "is_name": true, "origin": "Britain"},  {"name": "Ram Kumar", "is_name": true, "origin": "India"}, {"name": "dfgdf KfnqDuxw", “is_name”: false, "origin": "NA"}]
"""

    self.raa = RAA(sys_prompt=self.system_prompt)    # chatgpt completion API wrapped up as an agent

  def predict(self, names: List[str]) -> List[Dict[str, str]]:
    """
    names: list of names to predict

    The 1st turn shows assistant a task example
    """
    messages = [{"role": "user", "content": self.user_prompt_turn_1}]
    messages += [{"role": "assistant", "content": self.ai_response_turn_1}]
    messages += [{"role": "user", "content": ', '.join(names)}]

    predictions_str = self.raa.get_completion_from_messages(messages, temperature=0.2, max_tokens=1000, debug=True)

    # Expected return is a json formatted answer
    # Verify if predictions_str parsed to a dict (i.e. valid json)
    try:
      predictions = json.loads(predictions_str)
    except:
      raise ValueError(f'Malformed json response: {predictions_str}')
      
    return predictions
  
  def print_prompt(self, inputs: List[str]) -> str:
    return self.system_prompt + "\n" + self.user_prompt_turn_1 + "\n" + self.ai_response_turn_1 + "\n" + ', '.join(inputs)


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
        "If the message is a job application, resume, or related to seeking employment at the real estate agency (even if it mentions real estate), classify it as SPAM.",
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

    # Expected return is a json formatted answer
    # Verify if predictions_str parsed to a dict (i.e. valid json)
    try:
      predictions = json.loads(predictions_str)
    except:
      raise ValueError(f'Malformed json response: {predictions_str}')

    return predictions
  
  def recover_json(self, s: str):
    """
    Extract the largest valid JSON substring from s.
    Expected strucutre is a list of dict with keys "Prediction" and "Reasoning"

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

  def print_prompt(self, inputs: List[str]) -> str:
    inputs = '\n'.join([f"<input>{text}</input>" for text in inputs])
    user_prompt = f"""
    Can you please classify the following messages delimited by <inputs> and </inputs>? 
<inputs>
{inputs}
</inputs>
please provide predictions for each with either "NOT_SPAM", "SPAM", "TEST" or "NOT_SURE" together with your reasonings in less than 20 words 
in the format of json that is a list of dict with keys "Prediction" and "Reasoning". Provide only the JSON output without any additional text or explanation.
"""
    return self.system_prompt + "\n" + user_prompt + "\n"
    

class DataAugmentationAgent:
  def __init__(self, system_prompt: str = None, num_of_takes = 5, max_words=50):
    self.system_prompt = system_prompt
    self.num_of_takes = num_of_takes
    self.max_words = max_words

    if system_prompt is None:
      self.system_prompt = f"""You are a data augmentation agent able to take a number of sample messages and 
generate {self.num_of_takes} semantically similar variations of each. It is important to randomize any PII 
you encounter but without redacting. Limit each to less than {self.max_words} words. Input messages are delimited like <inputs><input>
message_0</input><input>message_1</input> etc. etc. </inputs> and you should respond in json format. Please note that each input message 
should be used as the key in the JSON response, and under each key, provide 5 semantically similar variations of that message.
Whenever 'NA' (Not Available) is encountered in a message, it should not be changed, as it carries semantic significance.
Please ensure that any special characters, especially quotation marks, within the JSON values are properly escaped 
using a backslash (e.g., \"). This will ensure the output remains valid JSON format and is parsable. 

{{"message_0": ["variation_0", "variation_1", ... ],
 "message_1": ["variation_0", "variation_1", ... ],
...
}}

Note: "message_0", "message_1" are independent of each other, and so their own set of variations should be as dissimilar from
each other as message_0 is from message_1.
"""

    self.raa = RAA(sys_prompt=self.system_prompt)    # chatgpt completion API wrapped up as an agent

  def predict(self, texts: List[str]) -> Dict[str, List[str]]:
    user_prompt = self._format_input(texts)
    user_prompt += "Provide only the JSON output without any additional text or explanation."

    messages = [{"role": "user", "content": user_prompt}]
    predictions_str = self.raa.get_completion_from_messages(messages, temperature=0.1, max_tokens=500, debug=False)

    # Expected return is a json formatted answer
    # Verify if predictions_str parsed to a dict (i.e. valid json)
    try:
      predictions = json.loads(predictions_str)
    except:
      raise ValueError(f'Malformed json response: {predictions_str}')
      
    return predictions
  
  def print_prompt(self, inputs: List[str]) -> str:
    user_prompt = self._format_input(inputs) 
    user_prompt += "Provide only the JSON output without any additional text or explanation."
    return self.system_prompt + "\n" + user_prompt
  
  def _format_input(self, inputs: List[str]) -> str:
    return "<inputs>\n" + '\n'.join([f"<input>{text}</input>" for text in inputs]) + "\n</inputs>\n"
    


    


def fast_extract_valid_json(s: str):
  """
  Extract valid JSON substring from s.

  This 
  """
  raise NotImplemented("This function needs more testing and debugging.")
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

