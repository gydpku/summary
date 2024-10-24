from vllm import LLM, SamplingParams
import torch
import pdb
import sqlite3
#from openai_call import query_azure_openai_chatgpt_chat
def label_transform(label):
    if label==1:
        return 'neutral'
    if label==0:
        return 'entailment'
    if label==2:
        return 'contradiction'
sampling_params = SamplingParams(temperature=0.0,max_tokens=600, top_p=0.95)
def results_collect(model_path,valid_data,task):
    trained_model=LLM(model=model_path,gpu_memory_utilization=0.95)
    if task=='sql':
        failed_cases,correct_cases=sql_evaluation(trained_model,valid_data)
    elif task=='nli':
        failed_cases,correct_cases=nli_evaluation(trained_model,valid_data)
    del trained_model
    torch.cuda.empty_cache()
    return failed_cases,correct_cases

def nli_evaluation(trained_model,valid_data):
    id=0
    failed_cases=[]
    correct_cases=[]
    for data in valid_data:
        prompt=data['Input']
        output=trained_model.generate(prompt, sampling_params)
        predicted_res=output[0].outputs[0].text
        label=label_transform(data['Output'])
        if label not in predicted_res:
            failed_cases.append((id,prompt,predicted_res,label,data))
        else:
            correct_cases.append((id,prompt,predicted_res,label,data))
        id+=1
    #id,prompt,prior_pred+predicted_sql,valid_data[id],ground_truth,predicted_res,ground_truth_res
    return failed_cases,correct_cases
def sql_evaluation(trained_model,valid_data):
    id=0
    failed_cases=[]
    correct_cases=[]
    for triple in valid_data:
        
        db_id,prompt,ground_truth=triple
        prompt=prompt.replace('SELECT','')
        db_path='/dccstor/obsidian_llm/yiduo/AgentBench/DAMO-ConvAI/bird/data/train/train_databases/{0}/{0}.sqlite'.format(db_id)
        prompt+=' To generate the SQL query to' #print(db_path) #pdb.set_trace()
        conn = sqlite3.connect(db_path)
        output=trained_model.generate(prompt, sampling_params) #pdb.set_trace()
        predicted_sql = output[0].outputs[0].text
        #pdb.set_trace()
        prior_pred=predicted_sql.split('final SQL')[0]
        try:
            predicted_sql = predicted_sql.split('final SQL')[1].strip()
        except:
            predicted_sql = 'SELECT'+predicted_sql.split('SELECT')[1]
        predicted_sql=predicted_sql.split(';')[0]
        predicted_sql=predicted_sql[predicted_sql.find('SELECT'):] #[1:]
        cursor = conn.cursor()
    #    pdb.set_trace()
        try:
            cursor.execute(predicted_sql)
            predicted_res = cursor.fetchall()
            cursor.execute(ground_truth)
            ground_truth_res = cursor.fetchall()
    #print('results',predicted_res,'truth',ground_truth_res,'\n')
            if set(predicted_res) != set(ground_truth_res):
                failed_cases.append((id,prompt,prior_pred+predicted_sql,valid_data[id],ground_truth,predicted_res,ground_truth_res))
            else:
                correct_cases.append((id,prompt,prior_pred+predicted_sql,valid_data[id],ground_truth,predicted_res,ground_truth_res))
        except Exception as e:
            failed_cases.append((id,prompt,predicted_sql,valid_data[id],ground_truth,str(Exception)+str(e)))
        return failed_cases,correct_cases