from generate_data import *
from cases_collect import *
from valid_analysis import *
import subprocess

store_name = 'medical_data_5k'
task = 'nli'

data_group = inference_data(
    'The task is to generate medical inference data based on the provided medical passage.',
    domain='Medical',
    num=5000,
    store_name='medical_data_5k',
    sample_num=3,
    passage_num=5000,
    types=None
)

# Run the finetune.py script using subprocess
subprocess.run([
    'python', 'finetune.py',
    '--data_path', store_name,
    '--model_path', '/dccstor/obsidian_llm/yiduo/h100_data/llama-3-8b',
    '--output_path', './medical_model'
])

failed_cases, correct_cases = results_collect('./medical_model', data_group, 'nli')
IP_reason, OOA_reason = results_analysis(failed_cases, correct_cases)