from typing import List, Dict
from raa import RAA    
from tenacity import RetryError
import openai
try:
  from openai.error import OpenAIError, APIConnectionError
except:
  print("Not importing OpenAIError from openai.error, newer version has changed")
import json, re

from ..utils.misc import num_tokens_from_string, num_tokens_from_messages

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
               llm_model: str,
               system_prompt: str = None, 
               guidelines: List[str] = None, 

               use_n_shot: bool = False, 
               spam_n_shot_samples: List[str] = None,
               not_spam_n_shot_samples: List[str] = None,
               test_n_shot_samples: List[str] = None
               ):
    
    self.llm_model = llm_model
    self.system_prompt = system_prompt
    self.guidelines = guidelines

    self.use_n_shot = use_n_shot
    if self.use_n_shot:
      self.spam_n_shot_samples = spam_n_shot_samples
      self.not_spam_n_shot_samples = not_spam_n_shot_samples
      self.test_n_shot_samples = test_n_shot_samples

    self.inputs = None    # placeholder for prediction inputs

    if self.guidelines is None:   # default available
      self.guidelines = [
        "If the message is related to buying, selling, renting, or inquiring about a property, then it is NOT_SPAM.",
        "If the message concerns home valuation, it is NOT_SPAM",
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

    if self.use_n_shot:
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
"""
      if self.use_n_shot:
        self.system_prompt += f"""

Here are a few examples for you to learn:
{n_shot_sample_for_spam}
{n_shot_sample_for_not_spam}
{n_shot_sample_for_test}
"""

    self.raa = RAA(llm_model=self.llm_model, sys_prompt=self.system_prompt)    # chatgpt completion API wrapped up as an agent


  def predict(self, texts: List[str], temperature=0.1, debug=False) -> List[Dict]: 

    messages = self.construct_openai_user_prompt(texts)
    predictions_str = self.raa.get_completion_from_messages(messages, temperature=temperature, max_tokens=500, debug=debug)

    # Expected return is a json formatted answer
    # Verify if predictions_str parsed to a dict (i.e. valid json)
    try:
      predictions = json.loads(predictions_str)
    except:
      raise ValueError(f'Malformed json response: {predictions_str}')

    return predictions
  
  def calc_cost_estimate(self, texts: List[str]) -> int:
    messages = self.construct_openai_user_prompt(texts)
    messages.insert(0, {"role": "system", "content": self.system_prompt})
    n_tokens = num_tokens_from_messages(messages)

    min_cost = n_tokens/1000 * 0.004
    max_cost = n_tokens/1000 * 0.12 
    return n_tokens, round(min_cost, 4), round(max_cost, 4)

  
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
    messages = self.construct_openai_user_prompt(inputs)
    user_prompt = messages[-1]['content']

    return self.system_prompt + "\n" + user_prompt + "\n"
    
  def construct_openai_user_prompt(self, inputs: List[str]) -> List[Dict]:
    """
    Construct messages to send to openai
    """
    inputs = '\n'.join([f"<input>{text}</input>" for text in inputs])
    user_prompt = f"""
    Can you please classify the following messages delimited by <inputs> and </inputs>? 
<inputs>
{inputs}
</inputs>
please provide predictions for each with either "NOT_SPAM", "SPAM", "TEST" or "NOT_SURE" together with your reasonings in less than 20 words 
in the format of json that is a list of dict with keys "Prediction" and "Reasoning". Provide only the JSON output without any additional text or explanation.
"""
    messages = [{"role": "user", "content": user_prompt}]

    return messages

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
    

class LocalLogicGPTRewriter:
  def __init__(self, 
               llm_model: str, 
               available_sections = ['housing', 'transport', 'services', 'character'], 
               property_type: str = None,
               transaction_type: str = 'SALE',
               include_start_with_guideline: bool = True
               ):
    self.llm_model = llm_model
    self.available_sections = available_sections.copy()  
    self._sections = available_sections.copy()
    self.property_type = property_type
    self.transaction_type = transaction_type
    self.include_start_with_guideline = include_start_with_guideline

    self.property_pluralized = None    
    if self.property_type is not None:
      if self.property_type.lower() == 'condo':
        self.property_pluralized = "Condos"
      elif self.property_type.lower() == 'townhouse':
        self.property_pluralized = "Townhouses"
      elif self.property_type.lower() == 'semi-detached':
        self.property_pluralized = "Semi-detached Homes"
      elif self.property_type.lower() == 'luxury':
        self.property_pluralized = "Luxury Homes"
      elif self.property_type.lower() == 'investment':
        self.property_pluralized = "Investment Properties"
      elif self.property_type.lower() == 'rental':
        self.property_pluralized = "Residential Rental Properties"
      else:
        self.property_pluralized = f"{self.property_type.capitalize()} Homes"

    self.property_pluralized_used_in_start_with = None
    if self.property_type is not None:
      if self.property_type.lower() != 'rental':
        self.property_pluralized_used_in_start_with = self.property_pluralized + " for sale"
      elif self.property_type.lower() == 'rental':
        self.property_pluralized_used_in_start_with = "Apartments, Condos and Houses for rent"
      else:
        self.property_pluralized_used_in_start_with = self.property_pluralized + " for sale"

    self._construct_sys_prompt()

    # chatgpt completion API wrapped up as an agent
    self.raa = RAA(
      llm_model=self.llm_model, 
      sys_prompt=self.system_prompt, 
      # api_key_env_var='FAKE_JUMPTOOLS_OPENAI_API_KEY'
      api_key_env_var='JUMPTOOLS_OPENAI_API_KEY'
    )    

  @property
  def sections(self):
    return self._sections

  @sections.setter
  def sections(self, value):
    self._sections = value
    self._construct_sys_prompt()

  def test_openai_health(self) -> bool:
    return self.raa.test_health()

  def rewrite(self, params_dict: Dict[str, str] = None, use_rag=True, **kwargs) -> Dict[str, str]:
    """
    params_dict: contains info for RAG, numeric information to be injected into user prompt. Use it if provided.
    """
    error_message = ''
    prediction_str = None

    messages = self.construct_openai_user_prompt(params_dict=params_dict, use_rag=use_rag, **kwargs)
    try:
      prediction_str = self.raa.get_completion_from_messages(messages, temperature=1.0, max_tokens=1000, debug=False)
      # parse housing, transport, services, character from prediction_str
      # simulate invalid prediction_str 
      # prediction_str = "corrupted junk"
      # prediction_str = "<housing>This area is really awesome to live in.</housing>"
      parsed_response = self.parse_response(prediction_str)

      # rewrite is considered successful only if all sections are being rewritten in the required format processed in parse_response,
      # so check this here.
      success = True
      for section in self._sections:
        rewrite = parsed_response.get(section, None)
        if rewrite is None or len(rewrite) < 5:    # TODO: just some text for now.
          success = False
          break

      if not success:
        error_message += f'Not all sections are rewritten. Please try again. '

    except RetryError as e:
      # print(f'RetryError caught in LocalLogicGPTRewriter.rewrite(): {e}')
      
      last_attempt = e.last_attempt   # need to unwrap the exceptions
      # print(type(last_attempt.exception()))

      if last_attempt:
        if self.raa.openai_version == ">=1.3.0":
          try:
            last_attempt.result()
          except openai.APIError as ae:
            error_message += str(ae)
            if hasattr(ae, 'error'):
              error_message += f" OpenAI Error: {ae.error.get('message', None)}" 
            if hasattr(ae, 'chat_messages'):
              error_message += f" Chat messages: {ae.chat_messages}"
          except openai.APIConnectionError as ace:
            error_message += str(ace)
            if hasattr(ace, 'chat_messages'):
              error_message += f" Chat messages: {ace.chat_messages}"
          except openai.APIStatusError as ase:
            error_message += str(ase)
            if hasattr(ase, 'chat_messages'):
              error_message += f" Chat messages: {ase.chat_messages}"
          except Exception as e:
            print(f'caught a generic exception: {e}')
            error_message += str(e)
        else:
          try:
            last_attempt.result()
          except OpenAIError as oe:
            error_message += str(oe)
            if hasattr(oe, 'error'):
              error_message += f" OpenAI Error: {oe.error.get('message', None)}" 
            if hasattr(oe, 'chat_messages'):
              error_message += f" Chat messages: {oe.chat_messages}"
          except Exception as e:
            print(f'caught a generic exception: {e}')
            error_message += str(e)
    
    except Exception as e:      
      error_message += str(e)
      print(f'Exception caught in LocalLogicGPTRewriter.rewrite(): {error_message}')
    finally: 
      # return the dict that will contain the consolidated error messages
      if error_message != '':
        return {**{key: None for key in kwargs.keys()}, "error_message": error_message}
      else:
        parsed_response['error_message'] = None
        return parsed_response


  def _construct_sys_prompt(self):
    tag_for_sections = ', '.join([f"<{section}></{section}>" for section in self._sections])
    are_or_is, subject_verb, plural_section = ("are", "They are", "sections") if len(self._sections) > 1 else ("is", "It is", "section")

    self.system_prompt = f"""
You are locallogic writer with lot of experience working with neighborhood profiles. 
Please rewrite the below with interesting variations keeping the tone and style similar. 
There {are_or_is} {len(self._sections)} {plural_section} and {subject_verb.lower()} delimited by {tag_for_sections}.
"""

  def construct_openai_user_prompt(self, params_dict: Dict[str, str] = None, use_rag=True, **kwargs) -> List[Dict]:
    """
    Construct messages to send to openai
    """
    if self.property_type is not None:
      assert params_dict is not None, "if rewrite is property specific, at least one dynamic param must be provided."

    # Figure out what sections are provided and set sections to them 
    # (which consequentially update the sys prompt as appropriate).
    self.sections = [section for section in self.available_sections if kwargs.get(section, None) is not None]

    if use_rag:
      rag_string = None
      if params_dict is not None:
        params_dict_str = '\n  '.join([f"{k+1}) {key}: {value}" for k, (key, value) in enumerate(params_dict.items()) if value is not None])
        if self.property_type is None:
          rag_string = f"""For housing section only if present, please incorporate these data points:

  {params_dict_str}

"""
        else:
          rag_string = f"""For housing section only if present, rewrite text customized for {self.property_pluralized} in the city referenced
while utilizing these data points:

  {params_dict_str}

"""
          
    else:  # placeholder way
      placeholder_string = "Anything bracketed by [] are placeholders to be determined, please retain them without alteration.\n"
      if params_dict is not None:
        param_keys = list(params_dict.keys())
        # params_dict_str = '\n'.join(param_keys)
        params_dict_str = '\n  '.join([f"{k+1}) {key} which is {value}" for k, (key, value) in enumerate(params_dict.items()) if value is not None])
        if self.property_type is None:
          placeholder_string += f"""For housing section only if present, please incorporate these infos using the following placeholders:
  
  {params_dict_str}

"""
        else:
          placeholder_string += f"""For housing section only, rewrite text customized exclusively for {self.property_pluralized} in the city referenced. 
For any stats and numeric info mentioned in the original text, provide a version specific to {self.property_pluralized} using these placeholders: 

  {params_dict_str} 

For stats and numeric info that are agnostic to property type {self.property_type.lower()}, please retain.
"""

    section_strings = []
    for section in self._sections:
      content = kwargs.get(section, None)
      section_string = f"<{section}>{content}</{section}>" if content is not None else "" #f"<{section}>None</{section}>"
      section_strings.append(section_string)

    # Add guidelines
    if self.property_type is None:
      guidelines = f"""

IMPORTANT

0. Wrap each section in its own tag, e.g. <housing></housing>
1. Ensure the rewrite is in the same language as the original content."""
      if self.include_start_with_guideline:
        guidelines += f"""
2. For housing if present, always start with 'Homes for sale in {{whatever city}}'.""" if self.transaction_type == 'SALE' else f"""
2. For housing if present, always start with 'Houses and Condos for rent in {{whatever city}}'."""
      guidelines += f"""
3. output something with roughly the same # of words each. 
4. Verbiage may not be too "flowery"
5. Be sure to retain numerical and highway or major street info.
"""      
    else:    
      guidelines = f"""

IMPORTANT

0. Wrap each section in its own tag, e.g. <housing></housing>
1. Ensure the rewrite is in the same language as the original content."""
      if self.include_start_with_guideline:
        guidelines += f"""
2. For housing if present, always start with '{self.property_pluralized_used_in_start_with} in {{whatever city}}', and be sure to retain {self.property_type.lower()} agnostic info without referring to {self.property_type.lower()}."""
      guidelines += f"""
3. output something with roughly the same # of words each. 
4. Verbiage may not be too "flowery"
5. Be sure to retain numerical and highway or major street info.
"""      

    if use_rag:
      user_prompt = (rag_string if rag_string is not None else "") + "\n".join(section_strings) + guidelines
    else:
      user_prompt = placeholder_string + "\n".join(section_strings) + guidelines

    messages = [{"role": "user", "content": user_prompt}]
    return messages
    
    
  def print_prompt(self, **kwargs) -> str:
    messages = self.construct_openai_user_prompt(**kwargs)
    user_prompt = messages[-1]['content']   # get the user role message content

    return self.system_prompt + "\n" + user_prompt + "\n"
  
  def parse_response(self, prediction_str: str) -> Dict[str, str]:
    """
    Parse prediction_str into housing, transport, services, character
    """

    parsed_response = {}

    for section in self._sections:
      # Construct regex pattern for each section
      pattern = re.compile(f'<{section}>(.*?)</{section}>', re.DOTALL)

      # Search for the pattern in the input_text
      match = pattern.search(prediction_str)
      
      if match:
        # If a match is found, store the content in the parsed_response dictionary
        parsed_response[section] = match.group(1).strip()
      else:
        # If no match is found, store None in the parsed_data dictionary
        parsed_response[section] = None

    return parsed_response



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

