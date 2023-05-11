from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
model.to("cuda")


def translate(article):
    inputs = tokenizer(article, return_tensors="pt")
    inputs = inputs.to("cuda")

    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id["arb_Arab"],
        max_length=len(article.split(" ")) * 2,
    )
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]


with open("./data/data_pubmed_simplified.json", "r") as f:
    data = json.load(f)

translated_data = []


for d in tqdm(data):
    d.update(
        {
            "ar_text": translate(d["text"]),
            "ar_simplified": translate(d["simplified"]),
        }
    )
    translated_data.append(d)

with open("./data/data_pubmed_simplified_ar.json", "w") as f:
    json.dump(translated_data, f, indent=4)
