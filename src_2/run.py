from generate_data import *
from cases_collect import *
from valid_analysis import *
import subprocess
import argparse
import torch
import os
import pdb
import time
parser = argparse.ArgumentParser(description="Run interactive_large_data.sh with parameters")
parser.add_argument("--store_name", type=str, default="medical_data_5k_1", help="Name of the data store")
parser.add_argument("--task", type=str, default="nli", help="Task type")
parser.add_argument("--task_instruction", type=str, default="The task is to generate medical inference data based on the provided medical passage.", help="Task instruction")
parser.add_argument("--model_name", type=str, default="medical_5k_1_model", help="Model name")
parser.add_argument("--domain", type=str, default="Medical", help="Domain")
parser.add_argument("--num", type=int, default=5000, help="data number")
parser.add_argument('--path', type=str, help="Path to the data",default='/dccstor/obsidian_llm/yiduo/summary/src/')
parser.add_argument('--model_path', type=str, help="Path to the model",default='/dccstor/obsidian_llm/yiduo/h100_data/llama-3-8b')
parser.add_argument('--output_path', type=str, help="Path for output",default='/dccstor/obsidian_llm/yiduo/copy_v2/finetuned_models/')

#data_path = os.path.join(args.path, args.store_name)
#output_path = os.path.join(args.path, args.model_name)
args = parser.parse_args()
data_group = data_sample(
    args.task_instruction,
    domain=args.domain,
    num=args.num,
    store_name=args.store_name,
    task_name=args.task,
    sample_num=3,
    passage_num=args.num,
    types=None
)
data_path = os.path.join(args.path, args.store_name)
output_path = os.path.join(args.path, args.model_name)
command = ["sh", "launch_distributed_finetune.sh", data_path, args.model_path, output_path]
# Run the finetune.py script using subprocess
process = subprocess.Popen(command)

# Wait for the process to complete by checking periodically
while process.poll() is None:
    print("Job still running... waiting for completion.")
    time.sleep(2)
# test evaluation
if args.task=='nli':
    test_examples=[]
    dataset=load_dataset('hippocrates/MedNLI_test')  
    #for id in range(len(multi_nli['train'])): examples.append('INPUT: '+'Premise: '+multi_nli['train'][id]['premise']+'Hypothesis: '+multi_nli['train'][id]['hypothesis']+'Output'+label_transform(multi_n$
#    instruction='The domain is Medical. The TASK: Please classify the relationship between the given premise and hypothesis into one of the following labels: entailment, contradiction, or neutral. Retur$
    for id in range(len(dataset['test'])):test_examples.append({'Input':dataset['train'][id]['query'],'Output':dataset['train'][id]['answer']})
#valid data collect
    f_test,c_test=valid_results_collect(output_path, test_examples, args.task)
    print(len(c_test)/(len(c_test)+len(f_test)),'acc')
    pdb.set_trace()
valid_data=torch.load('{0}_demo.pt'.format(args.task))
failed_cases, correct_cases = valid_results_collect(output_path, valid_data, args.task)
pdb.set_trace()
IP_reason, OOA_reason = results_analysis(failed_cases, correct_cases)
