# TEA-AgriVLN
# LLM API

from openai import OpenAI
import yaml


def llm_api(model, system_prompt, user_prompt, who):

   with open("key.yaml", "r") as f:
      key = yaml.safe_load(f)

   base_url = key['base_url']
   api_key = key['api_key'][who]

   client = OpenAI(api_key=api_key, base_url=base_url)

   completion = client.chat.completions.create(
      model=model,
      stream=False,
      messages=[
         {"role": "system", "content": system_prompt},
         {"role": "user", "content": user_prompt}
      ]
   )

   return completion.choices[0].message.content