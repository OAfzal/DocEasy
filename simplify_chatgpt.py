import openai
import os
from dotenv import load_dotenv, find_dotenv
import json
from tqdm import tqdm
import time

_ = load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")


def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,  # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]


with open("./prompt_templates.json") as f:
    prompt_templates = json.load(f)["prompts"]

with open("./data/data_pubmed_to_simplify.json", "r") as f:
    data = json.load(f)


gen_data = []
save_file_name = "./data/data_pubmed_simplified.json"

for doc in tqdm(data):
    for p_tmp in prompt_templates:
        prompt = p_tmp.format_map({"text": doc})
        sleep_time = 10
        while True:
            try:
                out = get_completion(prompt)
                break
            except:
                sleep_time *= 2
                time.sleep(sleep_time)
                continue

        gen_data.append({"text": doc, "simplified": out, "prompt": p_tmp})
        with open(save_file_name, "w") as f:
            json.dump(gen_data, f, indent=4)
