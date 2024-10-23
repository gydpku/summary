from vllm import LLM, SamplingParams
import torch
import pdb
import sqlite3
#from openai_call import query_azure_openai_chatgpt_chat
sampling_params = SamplingParams(temperature=0.0,max_tokens=600, top_p=0.95)
def results_collect(model_name):
    trained_model=LLM(model=model_name,gpu_memory_utilization=0.95)
    failed_cases=[]
    correct_cases=[]
    valid_data=torch.load('valid_data.pt')[:100]
    valid_data_cot=torch.load('valid_data_cot.pt')
    id=0
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
#        pdb.set_trace() #predicted_sql=predicted_sql.split(';')[0] if ';' in predicted_sql else predicted_sql
        #predicted_sql=predicted_sql.replace('\\n',' ').replace('\\','')
        #print('pred',predicted_sql)
        cursor = conn.cursor()
#    pdb.set_trace()
        try:
            cursor.execute(predicted_sql)
            predicted_res = cursor.fetchall()
            cursor.execute(ground_truth)
            ground_truth_res = cursor.fetchall()
    #print('results',predicted_res,'truth',ground_truth_res,'\n')
            if set(predicted_res) != set(ground_truth_res):
                failed_cases.append((id,prompt,prior_pred+predicted_sql,valid_data_cot[id],ground_truth,predicted_res,ground_truth_res))
            else:
                correct_cases.append((id,prompt,prior_pred+predicted_sql,valid_data_cot[id],ground_truth,predicted_res,ground_truth_res))
        except Exception as e:
            failed_cases.append((id,prompt,predicted_sql,valid_data_cot[id],ground_truth,str(Exception)+str(e)))
        id+=1
    del trained_model
    torch.cuda.empty_cache()
    return failed_cases,correct_cases
