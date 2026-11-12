# TEA-AgriVLN
# Decision-Maker

import ollama
import re
import os
import json
import yaml
import sys
import base64
from openai import OpenAI

from agrivln.for_json import get_stop_start_time, clean_format_stl_state, append_action
from prompt import decide1_system_prompt, decide2_system_prompt, decide_user_prompt
from tea.alarm import alarm
from vlm import vlm_api


def encode_image(image_path):
   with open(image_path, "rb") as image_file:
      return base64.b64encode(image_file.read()).decode('utf-8')


def extract(gpt_output):
   action_match = re.search(r"<action>\s*(.*?)\s*</action>", gpt_output, re.DOTALL)
   state_match = re.search(r"<state>\s*(.*?)\s*</state>", gpt_output, re.DOTALL)
   thought_match = re.search(r"<thought>\s*(.*?)\s*</thought>", gpt_output, re.DOTALL)
   result = {
      "action": action_match.group(1).strip() if action_match else None,
      "state": state_match.group(1).strip() if state_match else None,
      "thought": thought_match.group(1).strip() if thought_match else None
   }
   return result


def extract_state(text):
   match = re.search(r"NO\.(\d+)\s+changes from (\w+)\s+to (\w+)", text)
   if match:
      subtask_no = int(match.group(1))
      from_state = match.group(2)
      to_state = match.group(3)
      return subtask_no, from_state, to_state
   else:
      return None, None, None
   

def update_subtask_state(new_STL, number, old_state, new_state):
   for subtask in new_STL:
      if subtask.get("step") == number:
         current_state = subtask.get("state")
         if current_state == old_state:
            subtask["state"] = new_state
            print(f"[INFO] Subtask {number}: {old_state} -> {new_state}.")
         else:
            print(f"[WARNING] Subtask {number} not updated: current state is '{current_state}', expected '{old_state}'.")
         return new_STL
   print(f"[ERROR] Subtask with step number {number} not found.")
   return new_STL


def append_stl_state(state_path, new_entry):
   try:
      with open(state_path, 'r') as f:
         data = json.load(f)
   except FileNotFoundError:
      data = []
   data.append(new_entry)
   with open(state_path, 'w') as f:
      json.dump(data, f, indent=3, separators=(",", ": "))


def load_json(path):
   with open(path, 'r') as f:
      return json.load(f)
   

def merge_stl_and_state(stl_path, state_path, current_time):
   STL = load_json(stl_path)
   STL_states = load_json(state_path)

   def time_str_to_float(time_str):
      minutes, tenths = time_str.split("'")
      return int(minutes) + int(tenths) / 10

   current_time_val = time_str_to_float(current_time)
   closest_state = None
   for state_snapshot in STL_states:
      snapshot_time = time_str_to_float(state_snapshot['time'])
      if snapshot_time <= current_time_val:
         closest_state = state_snapshot
      else:
         break
   if closest_state is None:
      raise ValueError(f"No matching STL state found for time {current_time}")
   state_dict = {s['step']: s['state'] for s in closest_state['subtask_list']}
   merged_STL = []
   for subtask in STL:
      step = subtask['step']
      merged_subtask = dict(subtask)
      merged_subtask['state'] = state_dict.get(step, "unknown")  # fallback
      merged_STL.append(merged_subtask)
   return merged_STL


def decide(exp, place, id, if_token, cache):

   with open("config.yaml", "r") as f:
      config = yaml.safe_load(f)
   
   FPS = config['decide']['FPS']
   t_b_interval = config['decide']['t_b_interval']
   safe_redundancy = config['decide']['safe_redundancy']
   temperature = config['decide']['temperature']
   vlm_mode = config['decide']['vlm_mode']

   t_a = 0
   t_b = 0
   predict_path = f"runs/{exp}/{place}_{id}/predict.json"

   # update 2026.7.13: cache for decision-making
   if cache == True:
      if os.path.exists(predict_path):
         with open(predict_path, "r", encoding="utf-8") as f:
            p = json.load(f)
         if len(p) > 0:
            t_a, t_b = map(int, p[-1]['time'].split("'"))
            t_b += t_b_interval
            # debug 2026.7.16
            if t_b == 10:
               t_b = 0
               t_a += 1
            print('[INFO] Cache for decision-making is activated.')

   label_path = f"A2A/{place}_{id}/label.json"
   with open(label_path, "r") as f:
      labels = json.load(f)

   stop_start_time = get_stop_start_time(labels)
   print(f"[INFO] Label STOP begins at: {stop_start_time}")
   max_time = float(stop_start_time) + (1 / FPS) * safe_redundancy
   print(f'[INFO] Max time step: {max_time}')

   stop_quantity = 0

   while t_a < max_time:

      t = f'{t_a}\'{t_b}' # time

      STL_path = f"runs/{exp}/{place}_{id}/STL.json"
      log_path = f"runs/{exp}/{place}_{id}/log.json"
      TEA_path = f"runs/{exp}/{place}_{id}/TEA_{t}.json"
   
      STL = merge_stl_and_state(STL_path, log_path, current_time=t)

      image_path = f"A2A/{place}_{id}/frames/frame_{t}.jpg"

      # reasoning 1
      if vlm_mode == 'local':
         messages = [
            {
               'role': 'system',
               'content': decide1_system_prompt()
            },
            {
               'role': 'user',
               'content': decide_user_prompt(STL),
               'images': [image_path]
            }
         ]
         response = ollama.chat(
            model=config['decide'][f'vlm_{vlm_mode}'], 
            messages=messages, 
            options={"temperature": temperature},
            think=False)
         message = response['message']['content']
      # update 2026.7.16: vlm can also be accessed by api
      elif vlm_mode == 'api':
         message = vlm_api(
            config['decide'][f'vlm_{vlm_mode}'], 
            decide1_system_prompt(), 
            decide_user_prompt(STL),
            [image_path],
            temperature,
            'decide'
            )
      else:
         sys.exit('[ERROR] The VLM mode is invalid.')

      # update 2025.11.4
      if if_token == 'True':
         
         token_prompt = response.prompt_eval_count
         token_completion = response.eval_count

         token_path = f"runs/{exp}/{place}_{id}/token.json"

         # Create directory if it doesn't exist
         os.makedirs(os.path.dirname(token_path), exist_ok=True)

         # If file exists, load it; otherwise start a new list
         if os.path.exists(token_path):
            with open(token_path, "r") as token_f:
               try:
                  data = json.load(token_f)
               except json.JSONDecodeError:
                  data = []
         else:
            data = []

         # Append the new record
         data.append({
            "place": place,
            "time": t,
            "token_prompt": token_prompt,
            "token_completion": token_completion
         })

         # Save back to JSON
         with open(token_path, "w") as token_f:
            json.dump(data, token_f, indent=4)

      result = extract(message)
      action = result['action']
      thought = result['thought']
      state = result['state']

      action_1 = action
      thought_1 = thought

      # update 2026.7.13: the TEA module
      a = alarm(action, f'{place}_{id}', f'frame_{t}')
      if a == 'on':
         # reasoning 2
         traversability_map_path = f"after_classifier/{place}_{id}/frame_{t}.png"
         if vlm_mode == 'local':
            messages = [
               {
                  'role': 'system',
                  'content': decide2_system_prompt()  # insert a description
               },
               {
                  'role': 'user',
                  'content': decide_user_prompt(STL),
                  'images': [image_path, traversability_map_path]  # additional image input
               }
            ]
            response = ollama.chat(
               model=config['decide'][f'vlm_{vlm_mode}'], 
               messages=messages, 
               options={"temperature": temperature},
               think=False)
            message = response['message']['content']
         elif vlm_mode == 'api':
            message = vlm_api(
               config['decide'][f'vlm_{vlm_mode}'], 
               decide2_system_prompt(), 
               decide_user_prompt(STL),
               [image_path, traversability_map_path],
               temperature,
               'decide'
               )
         else:
            sys.exit('[ERROR] The VLM mode is invalid.')

         action_before = action

         result = extract(message)
         action = result['action']
         thought = result['thought']
         state = result['state']

         action_2 = action
         thought_2 = thought

         # save
         data = {
            "exp": exp,
            "episode_id": f"{place}_{id}",
            "t": t,
            "TEA_effect": f"{action_1} -> {action_2}",
            "action_1": action_1,
            "thought_1": thought_1,
            "action_2": action_2,
            "thought_2": thought_2,
         }

         with open(TEA_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

         print(f"TEA on {t}: {action_before} -> {action}.")
      else:
         print(f"TEA off {t}.")

      append_action(predict_path, t, action, thought, state)

      # revision 2025.10.23 start
      if state == None:
         new_STL = STL
         print('[WARNING] State = None.')
      else:
      # revision 2025.10.23 end
         if 'keep' in state:
            new_STL = STL
         if 'change' in state:
            number, old_state, new_state = extract_state(state)
            new_STL = update_subtask_state(STL, number, old_state, new_state)

      print(f'{place}_{id}, {t_a}.{t_b}, {action}')

      t_b += t_b_interval
      if t_b == 10:
         t_b = 0
         t_a += 1

      new_STL_slimmed = [
         {"step": item["step"], "state": item["state"]}
         for item in new_STL
      ]
      next_STL_state = {
         "time": f"{t_a}'{t_b}",
         "subtask_list": new_STL_slimmed
      }
      append_stl_state(log_path, next_STL_state)
      clean_format_stl_state(log_path)
      
      if action == '[STOP]':
         stop_quantity += 1
      if stop_quantity >= 3:
         break
