# TEA-AgriVLN
# Backbone Homepage

import argparse
import time
import os
import json
import sys
import yaml

from backbone.STL import STL
from backbone.decide import decide
from agrivln.evaluate import evaluate


def check_label_format(label_path):
   valid_actions = {"[FORWARD]", "[LEFT ROTATE]", "[RIGHT ROTATE]", "[STOP]", "[WAIT]"}
   with open(label_path, "r") as f:
      labels = json.load(f)
   for i in range(len(labels)):
      entry = labels[i]
      if entry["action"] not in valid_actions:
         print(f"[ERROR] Action NO. {i} is invalid: {entry['action']}")
         return False
      if i < len(labels) - 1:
         end_time = round(labels[i]["time_range"][1], 3)
         next_start_time = round(labels[i + 1]["time_range"][0], 3)
         if end_time != next_start_time:
            print(f"[ERROR] Time steps {i} and {i+1} are not connected: {end_time} ≠ {next_start_time}")
            return False
   return True


if __name__ == '__main__':

   with open("config.yaml", "r") as f:
      config = yaml.safe_load(f)

   parser = argparse.ArgumentParser()
   parser.add_argument("-p", "--place", type=str, required=True, help="Place")
   parser.add_argument("-i", "--id_range", type=int, nargs='+', required=True, help="Input the range of ID")
   parser.add_argument("-t", "--if_token", type=str, required=False, default='False', help="Token calculation")
   args = parser.parse_args()
   place = args.place
   id_range = args.id_range
   if_token = args.if_token

   if len(id_range) == 1:
      id_range = id_range
   elif len(id_range) == 2:
      id_range = list(range(id_range[0], id_range[1] + 1))
   elif len(id_range) > 2:
      id_range = id_range
   else:
      print('[ERROR] Invalid ID range.')
      sys.exit(1)

   if if_token not in ['True', 'False']:
      print('[ERROR] Invalid if_token.')
      sys.exit(1)

   # running information
   exp = 'TEA-AgriVLN'
   llm_mode = config['subtask_list']['llm_mode']
   llm = config['subtask_list'][f'llm_{llm_mode}']
   vlm_mode = config['decide']['vlm_mode']
   vlm = config['decide'][f"vlm_{vlm_mode}"]

   print(f'[INFO] Experiment: {exp}')
   print(f'[INFO] LLM: {llm}')
   print(f'[INFO] VLM: {vlm}')
   print(f"[INFO] Place: {place}")
   print(f'[INFO] ID range: {id_range}')

   # run
   for id in id_range:
      dir_path = f"runs/{exp}/{place}_{id}"
      os.makedirs(dir_path, exist_ok=True)
      label_path = f"A2A/{place}_{id}/label.json"
      if check_label_format(label_path) == False:
         print(f'[ERROR] {place}_{id} label format is wrong.')
      else:
         print(f"[INFO] {place}_{id} label format is correct.")
         print(f'--- {place}_{id} starts ---')

         STL(exp, place, id, cache=True)
         time.sleep(0.1)

         decide(exp, place, id, if_token, cache=True)
         time.sleep(0.1)

         evaluate(exp, place, id)
         time.sleep(0.1)

         print(f'--- {place}_{id} ends ---')