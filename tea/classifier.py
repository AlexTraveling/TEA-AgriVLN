# TEA-AgriVLN
# Classifier

import ollama
import sys
import re
import os
import pickle
import yaml

from prompt import classifier_system_prompt
from vlm import vlm_api


def extract(message, tag):
   pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
   match = re.search(pattern, message, re.DOTALL)

   if match:
      return match.group(1).strip()
   else:
      return None


def classifier(place, id, name, image_path_list, output_dir, if_VLM):

   with open("config.yaml", "r") as f:
      config = yaml.safe_load(f)
      
   vlm_mode = config['classifier']['vlm_mode']
   robot_information = config['classifier']['robot_information']

   # skip if exist
   if os.path.isfile(f"{output_dir}/{name}.pkl") == True:
      print(f"{output_dir}/{name}.pkl exists.")
      return None

   if if_VLM == True:

      classifier_user_prompt = f'<robot_information> {robot_information} </robot_information>'

      if vlm_mode == 'local':
         messages = [
            {
               'role': 'system',
               'content': classifier_system_prompt()
            },
            {
               'role': 'user',
               'content': classifier_user_prompt,
               'images': image_path_list
            }
         ]
         response = ollama.chat(model=config['classifier']['vlm_local'], messages=messages)
         message = response['message']['content']

      # update 2026.7.17: the vlm for the classifier can also be accessed by api
      elif vlm_mode == 'api':
         message = vlm_api(
            config['classifier']['vlm_api'],
            classifier_system_prompt(),
            classifier_user_prompt,
            image_path_list,
            config['classifier']['temperature'],
            'classifier'
            )
         
      else:
         sys.exit('[ERROR] The VLM mode is invalid.')

      # save message
      with open(f"{output_dir}/{name}_message.txt", "w", encoding="utf-8") as f:
         f.write(message + "\n\n")

   # read message
   with open(f"{output_dir}/{name}_message.txt", "r", encoding="utf-8") as f:
      message = f.read()

   # extract
   thought_summary = extract(message, 'thought_summary')
   traversable_segment_list = extract(message, 'traversable_segments')
   non_traversable_segment_list = extract(message, 'non-traversable_segments')

   # debug 2026.7.2
   if non_traversable_segment_list == None:
      non_traversable_segment_list = extract(message, 'non_traversable_segments')

   # debug 2026.7.15: segment #n -> n
   traversable_segment_list = [int(re.search(r"\d+$", x.strip()).group()) for x in traversable_segment_list.split(",") if x.strip()]
   non_traversable_segment_list = [int(re.search(r"\d+$", x.strip()).group()) for x in non_traversable_segment_list.split(",") if x.strip()]
   
   # read pkl
   pkl_path = f'after_overlay/{place}_{id}/{name}.pkl'
   with open(pkl_path, "rb") as f:
      pkl_list = pickle.load(f)

   # update pkl
   for n in range(len(pkl_list)):
      pkl_list[n]["traversability"] = 'unknown'
   for i in traversable_segment_list:
      pkl_list[i-1]["traversability"] = 'yes'
   for i in non_traversable_segment_list:
      pkl_list[i-1]["traversability"] = 'no'
   for p in pkl_list:
      print(p['traversability'], end=' ')
   print()

   # save pkl
   with open(f'{output_dir}/{name}.pkl', "wb") as f:
      pickle.dump(pkl_list, f)
