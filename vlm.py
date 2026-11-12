# TEA-AgriVLN
# VLM API

from openai import OpenAI
import base64
import yaml


def encode_image(image_path):
   with open(image_path, "rb") as image_file:
      return base64.b64encode(image_file.read()).decode('utf-8')


def vlm_api(model, system_prompt, user_prompt, image_path_list, temperature, who):

   with open("key.yaml", "r") as f:
      key = yaml.safe_load(f)

   base_url = key['base_url']
   api_key = key['api_key'][who]
   
   client = OpenAI(api_key=api_key, base_url=base_url)

   user_content = [{
                  "type": "text",
                  "text": user_prompt
               }]
   for image_path in image_path_list:
      user_content.append({
                  "type": "image_url",
                  "image_url": {
                     "url": f"data:image/jpeg;base64,{encode_image(image_path)}",
                     "detail": "high"
                  }
               })

   completion_with_base64 = client.chat.completions.create(
      model=model,
      messages=[
         {
            "role": "system", 
            "content": system_prompt
         },
         {
            "role": "user",
            "content": user_content
         }
      ],
      temperature=temperature
   )
   
   return completion_with_base64.choices[0].message.content
