# TEA-AgriVLN
# Subtask List

import ollama
import json
import re
import os
import yaml
import sys

from agrivln.prompt import get_STL_system_prompt, get_STL_user_prompt
from llm import llm_api


def load_instruction_from_info(file_path):
   with open(file_path, 'r') as f:
      info = json.load(f)
   return info['instruction']


def save_subtask_list(message, output_path="STL.json"):
   content = message
   match = re.search(r"<subtask_list>\s*(\[.*?\])\s*</subtask_list>", content, re.DOTALL)
   if not match:
      print("[ERROR] Cannot find the content in <subtask_list> tag.")
      return
   try:
      subtask_list = json.loads(match.group(1))
   except json.JSONDecodeError as e:
      print("[ERROR] Fail to decode JSON:", e)
      return
   with open(output_path, "w") as f:
      json.dump(subtask_list, f, indent=3)
   print(f"[INFO] Subtask list is saved to {output_path}.")


def generate_initial_stl_state(stl_path, output_path):
   with open(stl_path, "r") as f:
      subtask_list = json.load(f)
   state = {
      "time": "0'0",
      "subtask_list": [
         {"step": subtask["step"], "state": "pending"} for subtask in subtask_list
      ]
   }
   with open(output_path, "w") as f:
      json.dump([state], f, indent=3)
   print(f"[INFO] Initial state of Subtask List is generated to: {output_path}.")


def STL(exp, place, id, cache):

   with open("config.yaml", "r") as f:
      config = yaml.safe_load(f)
   llm_mode = config['subtask_list']['llm_mode']

   STL_path = f"runs/{exp}/{place}_{id}/STL.json"
   log_path = f"runs/{exp}/{place}_{id}/log.json"

   # 2027.7.13 update: the cache mechanism
   if cache == True:
      if os.path.exists(STL_path) and os.path.exists(log_path):
         print('[INFO] Cache for STL is activated.')
         return None

   with open(f"A2A/{place}_{id}/info.json", 'r') as f:
      instruction = json.load(f)['instruction']

   if llm_mode == 'local':
      messages = [
         {
            'role': 'system',
            'content': get_STL_system_prompt()
         },
         {
            'role': 'user',
            'content': get_STL_user_prompt(instruction)
         }
      ]
      response = ollama.chat(model=config['subtask_list']['llm_local'], messages=messages)
      message = response['message']['content']
   
   # update 2026.7.18: llm can also be accessed by api.
   elif llm_mode == 'api':
      message = llm_api(
         config['subtask_list']['llm_api'],
         get_STL_system_prompt(),
         get_STL_user_prompt(instruction),
         'subtask_list'
      )
   
   else:
      sys.exit("[ERROR] The llm mode is invalid.")

   save_subtask_list(message, STL_path)
   generate_initial_stl_state(STL_path, log_path)

   return None
