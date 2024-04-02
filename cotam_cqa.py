import argparse
import json
import datasets
import numpy as np

import openai

from tqdm import tqdm
from constant import openai_key

client = openai.OpenAI(api_key=openai_key)
model_engine = "gpt-3.5-turbo-instruct"

def create_query(sentence, label_text, other_label_text):
     return f'''"{sentence}"
Please think step by step:
1. What are some other attributes of the above sentence except \"{label_text}\"?
2. How to write a similar sentence with these attributes and \"{other_label_text}\"?
3. Write such a sentence without any other explanation.'''

def decode_response(response):
    for line in response.split("\n"):
        if line.startswith("3."):
            return line[2:].strip().strip("\"")

def preprocess_data(data):
    items = dict()
    items['question'] = data['question']

    choices = data['choices']['text']
    items['choices'] = choices
    
    for j in range(len(choices)):
        if data['choices']['label'][j] == data['answerKey']:
            items['answer'] = choices[j]
            break
    return items

def attribute_manipulate(data):
    responses = [data]

    sentence = data['question']
    label_text = data['answer']
    other_label_texts = [c for c in data['choices'] if (c != label_text)]

    for other_label_text in other_label_texts:
        query = create_query(sentence, label_text, other_label_text)
        try:
            response = client.chat.completions.create(
                        model=model_engine,
                        temperature=0.,
                        messages=[
                            {'role': 'user', 'content': query},
                        ],
                        ).choices[0]['message']['content']
        except:
            response = ""
        other_sentence = decode_response(response)

        if other_sentence is not None:
            response = dict()
            response['question'] = other_sentence
            response['choices'] = data['choices']
            response['answer'] = other_label_text

            responses.append(response)
    return responses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--n_sample', type=int, default=2000)
    parser.add_argument('--n_sample', type=int, default=1)
    parser.add_argument('--seed_split', type=int, default=0)
    parser.add_argument('--seed_sample', type=int, default=0)

    args = parser.parse_args()

    dataset = datasets.load_dataset("tau/commonsense_qa")
    train_valid_split = dataset['train'].train_test_split(test_size=0.1, seed=args.seed_split)

    dataset_train = train_valid_split['train']
    dataset_test = train_valid_split['test']

    np.random.seed(seed=args.seed_sample)
    dataset_train = np.random.choice(dataset_train, args.n_sample, replace=False)
    with open(f'datasets/cqa_train_{args.n_sample}_{args.seed_split}_{args.seed_sample}.json', 'w') as file:
        for idx, data in tqdm(enumerate(dataset_train), total=len(dataset_train)):
            items = preprocess_data(data)
            responses = attribute_manipulate(items)
            for response in responses:
                json.dump(response, file)
                file.write('\n')

    with open(f'datasets/cqa_test_{args.n_sample}_{args.seed_split}_{args.seed_sample}.json', 'w') as file:
        for idx, data in tqdm(enumerate(dataset_test), total=len(dataset_test)):
            items = preprocess_data(data)
            json.dump(items, file)
            file.write('\n')