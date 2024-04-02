import argparse
import os.path as osp
import numpy as np
import torch
import json
import jsonlines

from datasets import load_dataset, Dataset
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoModelWithLMHead
from transformers import DataCollatorForSeq2Seq
from transformers import AutoTokenizer
from transformers.trainer_utils import set_seed

NUM_OPTIONS = 5
K = 10

def load_cqa(data_dir, complete=True):
    def gen_data(dataset):
        for data in dataset:
            for k, v in data.items():
                if "(response)" not in k:
                    yield {"input": v, "label": k}
    
    dataset = [items for items in jsonlines.open(osp.join(data_dir, 'csqa.cotam.train.jsonl')) if len(items) == 2 * NUM_OPTIONS - 1]
    dataset_train = Dataset.from_generator(gen_data, gen_kwargs={"dataset": dataset})
    dataset_train = dataset_train.shuffle()
    dataset_processed = dataset_train.train_test_split(test_size=((len(dataset_train) - K * NUM_OPTIONS * NUM_OPTIONS) / len(dataset_train)))

    if complete:            
        def prepare_input(example):
            example['input'] = example['question']
            example['label'] = example['answer']
            return example
        dataset_test = load_dataset('json', data_files=osp.join(data_dir, 'cqa_test.json'))['train']
        dataset_test = dataset_test.map(prepare_input,
                            remove_columns=['question', 'choices', 'answer', 'id', 'abstractive_explanation', 'extractive_explanation'])
        dataset_processed['train'] = dataset_train
        dataset_processed['test'] = dataset_test
    
    return dataset_processed

def load_cqa_synthesized(data_dir):
    def prepare_input(example):
        choices = example['choices']
        full_sample = example['question']
        for i in range(len(choices)):
            full_sample += f' ({chr(97 + i)}) {choices[i]}'
        example['input'] = full_sample
        example['label'] = example['answer']
        return example

    file_dict = {'train': osp.join(data_dir, 'cqa_train_2000_0_0.json'), 'valid': osp.join(data_dir, 'cqa_test_2000_0_0.json')}
    dataset = load_dataset('json', data_files=file_dict)
    dataset = dataset.map(prepare_input,
                          remove_columns=['question', 'choices', 'answer', 'response'])
    train_test_split = dataset['train'].train_test_split(test_size=0.1, seed=0)
    dataset['train'] = train_test_split['train']
    dataset['test'] = train_test_split['test']

    return dataset

def set_all_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)

    return f"Set all the seeds to {seed} successfully!"

def compute_metrics_text_aux(tokenizer):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        acc = np.mean(np.array(decoded_preds) == np.array(decoded_labels))

        return {'accuracy': acc}

    return compute_metrics

def compute_agreement(model_responses, true_responses):
    def compare_strings(str1, str2):
        str1 = str1.lower().replace(" ", "")
        str2 = str2.lower().replace(" ", "")

        return str1 == str2

    acc_list = []
    for i in range(len(true_responses)):
        count = 0
        for j in range(len(true_responses[i])):
            true_response = true_responses[i][j]
            model_response = model_responses[i][j]
            if compare_strings(true_response, model_response):
                count += 1
        acc = count / len(true_responses[i])
        acc_list.append(acc)

    return acc_list


def get_exp_dir(args):
    return f'{args.exp_name}'


class Model:
    def __init__(self, args):
        self.args = args
        # self.dataset = load_cqa(args.data_dir, args.complete)
        self.dataset = load_cqa_synthesized(args.data_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(args.from_pretrained)

        def tokenize_function(examples):
            model_inputs = self.tokenizer(
                examples['input'],
                max_length=args.max_input_length,
                truncation=True
            )

            with self.tokenizer.as_target_tokenizer():
                label_output_encodings = self.tokenizer(examples['label'], max_length=256, truncation=True)

            model_inputs['labels'] = label_output_encodings['input_ids']

            return model_inputs

        self.tokenized_datasets = self.dataset.map(
            tokenize_function,
            remove_columns=['input', 'label'],
            batched=True
        )

        self.metrics = compute_metrics_text_aux(self.tokenizer)
        if args.ckpt == 0:
            self.ckpt = None
        else:
            self.ckpt = args.ckpt
        self.iteration = -1
        self.load_path = args.from_pretrained

    def update_ckpt(self, ckpt):
        self.ckpt = ckpt
        self.load_path = self.get_model_path()

    def update_iteration(self):
        self.iteration += 1
        self.load_path = self.get_model_path()

    def get_model_path(self):
        return f'ckpts/{get_exp_dir(self.args)}/{self.args.seed}/{self.iteration}/checkpoint-{self.ckpt}'

    def train(self):
        seed = self.args.seed
        args = self.args

        exp_dir = get_exp_dir(args)
        output_dir = f'ckpts/{exp_dir}/{seed}/{self.iteration+1}'
        logging_dir = f'logs/{exp_dir}/{seed}/{self.iteration+1}'

        model = AutoModelWithLMHead.from_pretrained(self.load_path)
        self.update_iteration()

        training_args = Seq2SeqTrainingArguments(
            output_dir,
            remove_unused_columns=False,
            evaluation_strategy='steps',
            eval_steps=args.eval_steps,
            save_strategy='steps',
            save_steps=args.eval_steps,
            logging_dir=logging_dir,
            logging_strategy='steps',
            logging_steps=args.eval_steps,
            max_steps=args.max_steps,
            learning_rate=args.lr,
            gradient_accumulation_steps=args.grad_steps,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            predict_with_generate=True,
            seed=seed,
            data_seed=args.data_seed,
            local_rank=args.local_rank,
            bf16=args.bf16,
            generation_max_length=args.gen_max_len,
            prediction_loss_only=False,
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model='test_accuracy',
        )
        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=model)
        trainer_kwargs = {
            'model': model,
            'args': training_args,
            'train_dataset': self.tokenized_datasets['train'],
            'eval_dataset': {'test': self.tokenized_datasets['test'], },
            'data_collator': data_collator,
            'tokenizer': self.tokenizer,
            'compute_metrics': self.metrics,
        }

        trainer = Seq2SeqTrainer(**trainer_kwargs)
        trainer.train()

    def inference(self, dataset):
        model_path = self.load_path
        model = AutoModelWithLMHead.from_pretrained(model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model.to(device)
        response_lists = []
        for i in range(len(dataset)):
            response_list = []
            for j in range(len(dataset[0])):
                input_text = dataset[i][j]['input']
                input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(device)

                with torch.no_grad():  # if you're using PyTorch
                    output_ids = model.generate(input_ids, max_new_tokens=128)
                output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                response_list.append(output_text)

            response_lists.append(response_list)
        return response_lists



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='datasets/')
    parser.add_argument('--max_steps', type=int, default=10000)
    parser.add_argument('--eval_steps', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--optimizer_name', type=str, default='AdamW')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data_seed', type=int, default=None)
    parser.add_argument('--from_pretrained', type=str, default='google/t5-v1_1-base')
    parser.add_argument('--max_input_length', type=int, default=1024)
    parser.add_argument('--grad_steps', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--gen_max_len', type=int, default=64)
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--ckpt', type=int, default=0)
    parser.add_argument('--complete', action='store_true')

    args = parser.parse_args()

    set_all_seed(args.seed)
    model = Model(args)
    model.train()