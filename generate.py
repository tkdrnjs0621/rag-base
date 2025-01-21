from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse
import re
import time
from datetime import timedelta
from functools import partial
import logging
from datasets import load_dataset, Dataset
from tqdm import tqdm
import os
from torch.utils.data import DataLoader

from src.retriever import Retriever

def generate(model, tokenizer, dataloader, log_every, **kwargs):
    start = time.time()
    output_ids = []
    for i, inputs in tqdm(enumerate(dataloader, start=1),total=len(dataloader)):
        inputs = inputs.to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, pad_token_id=tokenizer.eos_token_id, **kwargs)
        output_ids.extend(outputs[:, inputs["input_ids"].size(1) :].tolist())
        if i % log_every == 0:
            end = time.time()
            elapsed = end - start
            total = elapsed * (len(dataloader) / i)
    return output_ids

def build_prompt(example, tokenizer):
    prompt = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=True)
    prompt_tokens = len(tokenizer(prompt, add_special_tokens=False).input_ids)
    return {"prompt": prompt, "prompt_tokens": prompt_tokens}

def collate_fn(batch, tokenizer):
    prompt = [example["prompt"] for example in batch]
    inputs = tokenizer(prompt, add_special_tokens=False, padding=True, return_tensors="pt")
    return inputs

def decode(example, tokenizer, feature):
    text = tokenizer.decode(example[feature + "_ids"], skip_special_tokens=True)
    return {feature: text}

def map_question_to_input(row,option,retriever):
    if(option=='naiive'):
        txt = row["question"]
        messages = [{'role':'system','content':'You are a helpful assistant that answer to the question.'},{"role":'user',"content":txt}]
        row["messages"]=messages
        return row
    elif(option=='rag'):
        txt = row["question"]
        rs = retriever.search_document(txt, top_n=5)
        txt2 = "[Knowledge]\n"+'\n'.join([t['title']+" "+t['text'] for t in rs])
        txt2 +='\n[Question]\n'
        txt2 += txt
        messages = [{'role':'system','content':'You are a helpful assistant that answer to the question with the given knowledge.'},{"role":'user',"content":txt2}]
        row["messages"]=messages
        return row
    
def map_entity(row):
    pattern = r"\(([^,]+),\s*([^\)]+)\)"

    ls = row["output"].strip().split('\n')
    row["entities"]=[re.sub(pattern, r"\1 (\2)",l.strip()) for l in ls]
    return row

def set_file_handler(logger, path, level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"):
    os.makedirs(os.path.dirname(path + "/run.log"), exist_ok=True)
    handler = logging.FileHandler(path + "/run.log")
    handler.setLevel(level)
    formatter = logging.Formatter(format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Full Run")

    ### Dataset Configuration
    parser.add_argument("--dataset_path", type=str, default="data/eval_data/triviaqa_test.jsonl", help="model name for evaluation")
    
    ### Saving Options
    parser.add_argument("--model_path", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="path to local peft adapter (e.g. peft/sft/lora/Llama-3.1-8B-Instruct)")
    parser.add_argument("--save_path", type=str, default="results/tqa_vanilla.jsonl", help="path to inference data to evaluate (e.g. inference/baseline/zero_v1/Llama-3.1-8B-Instruct)")
    parser.add_argument("--num_proc", type=int, default=16, help="number of processors for processing datasets")
    parser.add_argument("--log_every", type=int, default=20, help="logging interval in steps")
    
    parser.add_argument("--option",choices=['naiive','rag'],default='naiive')

    ### Generation Options
    parser.add_argument("--batch_size", type=int, default=16, help="batch size for inference")
    parser.add_argument("--max_tokens", type=int, default=300, help="generation config; max new tokens")
    parser.add_argument("--do_sample", type=bool, default=False, help="generation config; whether to do sampling, greedy if not set")
    parser.add_argument("--temperature", type=float, default=0.0, help="generation config; temperature")
    parser.add_argument("--top_k", type=int, default=50, help="generation config; top k")
    parser.add_argument("--top_p", type=float, default=0.1, help="generation config; top p, nucleus sampling")

    ### Retrieval Options
    parser.add_argument("--retrieval_model_name_or_path", type=str, default="facebook/contriever-msmarco", help="path to directory containing model weights and config file")
    parser.add_argument("--retrieval_embedding_size", type=int, default=768, help="path to directory containing model weights and config file")
    parser.add_argument("--passages", type=str, default='data/corpus/psgs_w100.tsv', help="Path to passages (.tsv file)")
    parser.add_argument("--passages_embeddings", type=str, default='data/corpus/wikipedia_embeddings/*', help="Glob path to encoded passages")
    parser.add_argument("--indexing_batch_size", type=int, default=1000000, help="Batch size of the number of passages indexed")
    parser.add_argument("--save_or_load_index", action="store_true", help="If enabled, save index and load index if it exists")
    parser.add_argument("--lowercase", action="store_true", help="If enabled, save index and load index if it exists")
    parser.add_argument("--normalize_text", action="store_true", help="If enabled, save index and load index if it exists")
    parser.add_argument("--retrieval_n_subquantizers", type=int, default=0, help="Number of subquantizer used for vector quantization, if 0 flat index is used")
    parser.add_argument("--retrieval_n_bits", type=int, default=8, help="Number of bits per subquantizer")
    parser.add_argument("--per_gpu_batch_size", type=int, default=1000000, help="Number of bits per subquantizer")
    parser.add_argument("--question_maxlength", type=int, default=100000, help="Number of bits per subquantizer")
    parser.add_argument("--max_k", type=int, default=10, help="Number of documents to retrieve per questions")
    
    args = parser.parse_args()


    dataset = Dataset.from_json(args.dataset_path)

    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2")
    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()


    if(args.option == 'naiive'):
        dataset = dataset.map(partial(map_question_to_input,option=args.option,retriever=None))
        dataset = dataset.map(partial(build_prompt, tokenizer=tokenizer),num_proc=args.num_proc)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_proc, collate_fn=partial(collate_fn, tokenizer=tokenizer), pin_memory=True) 
        print("Length of DataLoader :",len(dataloader))

        output_ids = generate(model, tokenizer, dataloader, args.log_every, max_new_tokens=args.max_tokens, do_sample=args.do_sample, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
        dataset = dataset.add_column("output_ids", output_ids) 
        dataset = dataset.map(partial(decode, tokenizer=tokenizer, feature="output"), num_proc=args.num_proc)
        dataset = dataset.map(map_entity)
        dataset.to_json(args.save_path, lines=True)
    elif(args.option =='rag'):
        retriever = Retriever(args)
        retriever.setup_retriever()
        dataset = dataset.map(partial(map_question_to_input,option=args.option,retriever=retriever))
        dataset = dataset.map(partial(build_prompt, tokenizer=tokenizer),num_proc=args.num_proc)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_proc, collate_fn=partial(collate_fn, tokenizer=tokenizer), pin_memory=True) 
        print("Length of DataLoader :",len(dataloader))

        output_ids = generate(model, tokenizer, dataloader, args.log_every, max_new_tokens=args.max_tokens, do_sample=args.do_sample, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
        dataset = dataset.add_column("output_ids", output_ids) 
        dataset = dataset.map(partial(decode, tokenizer=tokenizer, feature="output"), num_proc=args.num_proc)
        dataset = dataset.map(map_entity)
        dataset.to_json(args.save_path, lines=True)