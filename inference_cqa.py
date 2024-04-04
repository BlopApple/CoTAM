import os.path as osp
import argparse
import json

from datasets import load_dataset

from tune_cqa import Model, prepare_input

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='datasets/')
    parser.add_argument('--dataset', type=str, default='2000_0_0')
    parser.add_argument('--max_steps', type=int, default=10000)
    parser.add_argument('--eval_steps', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--optimizer_name', type=str, default='AdamW')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data_seed', type=int, default=None)
    parser.add_argument('--from_pretrained', type=str, required=True)
    parser.add_argument('--max_input_length', type=int, default=1024)
    parser.add_argument('--grad_steps', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--gen_max_len', type=int, default=64)
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--ckpt', type=int, default=0)
    parser.add_argument('--old', action='store_true')
    parser.add_argument('--complete', action='store_true')
    parser.add_argument('--subsample', type=int, default=-1)

    args = parser.parse_args()

    dataset_eval = load_dataset('json', data_files=osp.join(args.data_dir, 'cqa_test.json'))['train']
    dataset_eval = dataset_eval.map(prepare_input,
                            remove_columns=['question', 'choices', 'answer', 'id', 'abstractive_explanation', 'extractive_explanation'])
    
    model = Model(args)
    responses = model.inference(dataset_eval)

    with open(f'results/responses/{args.exp_name}_responses.json', 'w') as file:
        for response in responses:
            json.dump(response, file)
            file.write('\n')