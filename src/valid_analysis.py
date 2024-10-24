import sqlite3
import pdb
import re
from openai_call import query_azure_openai_chatgpt_chat

#from vllm import LLM, SamplingParams
import torch
import sqlite3
from datasets import concatenate_datasets
from datasets import load_from_disk
from generate_data import clean_and_collect_dataset,clean_output
from datasets import Dataset,DatasetDict
def concat_dataset(dataset1_name,data2,new_name):
    dataset_1=load_from_disk(dataset1_name)
    if isinstance(data2[0],str):
        datas=[eval(ele[ele.find('{'):ele.find('}')+1]) for ele in data2]
    else:
        datas=data2
    train_data=[{'instruction':data['input'],'output':clean_output(data['output'])} for data in datas]
    dataset_2=Dataset.from_dict({key: [str(dic[key]) for dic in train_data] for key in train_data[0]})
#    dataset_dict = DatasetDict({'train': dataset})
    dataset_2=dataset_2.shuffle(seed=2022)
    concatenated_dataset = concatenate_datasets([dataset_1, dataset_2])
    concatenated_dataset=concatenated_dataset.shuffle(seed=2022)
    concatenated_dataset.save_to_disk(new_name)
#    dataset_2=clean_and_collect_dataset(data2)

    
    
def results_collect(model_name):
    trained_model=LLM(model=model_name,gpu_memory_utilization=0.95)
    failed_cases=[]
    correct_cases=[]
    valid_data=torch.load('valid_data.pt')[:100]
    for triple in valid_data:
        db_id,prompt,ground_truth=triple
        db_path='/dccstor/obsidian_llm/yiduo/AgentBench/DAMO-ConvAI/bird/data/train/train_databases/{0}/{0}.sqlite'.format(db_id)
        #print(db_path) #pdb.set_trace()
        conn = sqlite3.connect(db_path)
        output=trained_model.generate(prompt, sampling_params) #pdb.set_trace()
        predicted_sql = output[0].outputs[0].text
        predicted_sql='SELECT'+predicted_sql.split('SELECT')[1]
        predicted_sql=predicted_sql.split(';')[0] if ';' in predicted_sql else predicted_sql
        predicted_sql=predicted_sql.replace('\\n',' ').replace('\\','')
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
                failed_cases.append((prompt,predicted_sql,ground_truth,predicted_res,ground_truth_res))
            else:
                correct_cases.append((prompt,predicted_sql,ground_truth,predicted_res,ground_truth_res))
        except Exception as e:
            failed_cases.append((prompt,predicted_sql,ground_truth,str(Exception)+str(e)))
    del trained_model
    torch.cuda.empty_cache()
    return failed_cases,correct_cases
            

def case_division(cases):
    OOA_cases=[]
    IP_cases=[]
    for id in range(len(cases)//5+1):
        cases_batch=cases[id*5:(id+1)*5]
        cases_text=[]
        case_id=0
        for case in cases_batch:
            if len(case)==4:
                case_text=''
                case_text+='Case {0}: Input:{1}\n'.format(case_id,case[0])
                case_text+='Model prediction:{0}\n'.format(case[1])
                case_text+='Label:{0}\n'.format(case[2])
                case_text+='Executing the prediction lead to this error:{0}\n'.format(case[3])
                cases_text.append(case_text)
                
            else:
                case_text=''
                case_text+='Case {0}: Input:{1}\n'.format(case_id,case[0])
                case_text+='Model prediction:{0}\n'.format(case[1])
                case_text+='Label:{0}\n'.format(case[2])
                case_text+='Executing the prediction lead to this result:{0}, but it is wrong, the correct result is:{1}\n'.format(case[3],case[4])
                cases_text.append(case_text)
            case_id+=1
        prompt="""There are two main types of failed cases: 
        1. 'Out of Ability' cases: These are instances where the model lacks the capability to solve the problem.
        2. 'Imprecise' cases: These are instances where the model has the capability but makes small errors in reasoning or data generation, resulting in an incorrect output (it may be similarto the label).
        One main difference is that the inference step of 'Out of Ability' cases are totally wrong or incompleted..
        For 'Imprecise' cases, their inference steps (CoT) are usually partially correct.
        Your task is to categorize the following failed cases into these two types.
        Failed cases:
        {cases}
        You directly and only output the categorized results in the following format:
        Out of Ability cases:
        [case id, case id, ...]
        Imprecise cases:
        [case id, case id, ...]
        """.format(cases=cases_text)
        division_result=query_azure_openai_chatgpt_chat(prompt)
        #pdb.set_trace()
        Out_of_ability_cases=division_result.split("Out of Ability" )[1].split("Imprecise")[0].strip()
        try:
            Imprecise_cases=division_result.split("Imprecise")[1].strip()
        except:
            continue #pdb.set_trace()
        numbers = re.findall(r'\d+', Out_of_ability_cases)
# Convert the extracted numbers to integers
        numbers = list(map(int, numbers))
        for number in numbers:
            try:
                OOA_cases.append(cases_text[number])
            except:
                continue #pdb.set_trace()
        numbers = re.findall(r'\d+', Imprecise_cases)
        numbers = list(map(int, numbers))
        for number in numbers:
            try:
                IP_cases.append(cases_text[number])
            except:
                continue
    return OOA_cases,IP_cases
def Induce_OOA_cases(OOA_cases):
    characteristics=None
    for id in range(len(OOA_cases)//1+1):
        cases_batch=OOA_cases[id*1:(id+1)*1]
        cases_batch=[case[case.find(':')+2:] for case in cases_batch]
        cases_batch=['Case {0}: '.format(id)+case for id,case in enumerate(cases_batch)]
        if not characteristics:
            prompt="""Your task is to analyze and identify the common characteristics of these challenging cases.
        Examples of such characteristics can include specific problem types, reasoning types, examples, and solutions etc.
        Note that this task is about data feature rather than model error analysis, you should not describle the model's error.
        The cases are:
        {OOA_cases}
        You should write them in the instruction format.
        Please directly output these common data characteristics, followed by steps for how to generate cases contain these characteristic.
        """.format(OOA_cases=cases_batch)
#            prompt="""How to generate many cases like this case? Your task is to summarize the general characteristics of the provided challenging case.
 #           The cases are:
  #      {OOA_cases}. You should write them in the instructions format.""".format(OOA_cases=cases_batch)
            characteristics2=query_azure_openai_chatgpt_chat(prompt)
            print(characteristics2)
            pdb.set_trace()
        else:
            prompt="""Your task is to analyze and identify 3~5 common data characteristics of these challenging cases.
            These characteristics are the core features that leading to the model inablity for processing the cases.
        Examples of such characteristics can include specific problem types, reasoning types, etc.
        Note that this task is about data feature rather than model error analysis, you should not describle the model's error.
        The cases are:
        {OOA_cases}
        The previous characteristics are:
        {characteristics}
        You need to refine the previous characteristics (generalization)or add the new characteristics to the previous ones.
        You should write them in the instruction format.
        Please directly and only output these common data characteristics.
        """.format(OOA_cases=cases_batch,characteristics=characteristics)
            characteristics=query_azure_openai_chatgpt_chat(prompt)
    #pdb.set_trace()
 #   prompt="The model is inable to solve the data with following charactristics:{0} Your task is to write data generation instructions for generating more data with these charactristics to improve the model. No specfic examples are needed. Instructions:{0}".format(characteristics)
    return characteristics #query_azure_openai_chatgpt_chat(prompt)
def Induce_IP_cases(IP_cases):
    criteria=None
    for id in range(len(IP_cases)//5+1):
        cases_batch=IP_cases[id*5:(id+1)*5]
        if not criteria:
            prompt="""Your task is to analyze and identify 3~5 common criteria for finding reasoning errors from these wrong cases.
            These criteria are the core rules that can use to identify the error pattern in the reasoning path of these cases.
        Examples of such criteria can include identification rules, etc.
        Note that this task is about evaluation criteria rather than model error analysis, you should not describle the model's error.
        The cases are:
        {OOA_cases}
        
        You should write them in the instruction format.
        Please directly output these evaluation criteria.
        """.format(OOA_cases=cases_batch)
            criteria=query_azure_openai_chatgpt_chat(prompt)
        else:
            prompt="""Your task is to analyze and identify 3~5 common criteria for finding reasoning errors from these wrong cases.
            These criteria are the core rules that can use to identify the error pattern in the reasoning path of these cases.
        Examples of such criteria can include identification rules, etc.
        Note that this task is about evaluation criteria rather than model error analysis, you should not describle the model's error.
        The cases are:
        {OOA_cases}
        The previous criteria are:
        {characteristics}
        You need to refine the previous criteria or add the new criteria to the previous ones.
        You should write them in the instruction format.
        Please directly and only output these common criteria.
        """.format(OOA_cases=cases_batch,characteristics=criteria)
            criteria=query_azure_openai_chatgpt_chat(prompt)
    #pdb.set_trace()
 #   prompt="The model is inable to solve the data with following charactristics:{0} Your task is to write data generation instructions for generating more data with these charactristics to improve the m$
    return criteria #characteristics #query_azure_openai_chatgpt_chat(prompt)
def Induce_Correct_cases(Correct_cases):
    characteristics=None
    for id in range(len(Correct_cases)//5+1):
        cases_batch=Correct_cases[id*5:(id+1)*5]
        cases_text=[]
        case_id=0
        for case in cases_batch:
            case_text=''
            case_text+='Case {0}: Input:{1}\n'.format(case_id,case[0])
            case_text+='Model prediction:{0}\n'.format(case[1])
            case_text+='Label:{0}\n'.format(case[2])
            case_text+='Executing the prediction lead to this correct result:{0}\n'.format(case[3])
            cases_text.append(case_text)
            case_id+=1
        if not characteristics:
            prompt="""Your task is to analyze and identify 3~5 common characteristics of these correct cases.
            These characteristics represent challenges that the current model is able to solve.
        Examples of such characteristics can include specific problem types, reasoning types, etc.

        The correct cases are:
        {Correct_cases}
        You should write them in the instruction format.
        Please directly output these common characteristics.
        """.format(Correct_cases=cases_text)
            characteristics=query_azure_openai_chatgpt_chat(prompt)
        else:
            prompt="""Your task is to analyze and identify 3~5 common characteristics of correct cases.
            These characteristics represent challenges that the current model is able to solve.
        Examples of such characteristics can include specific problem types, reasoning types, etc.

        The correct cases are:
        {Correct_cases}
        The previous characteristics are:
        {characteristics}
        You need to refine the previous characteristics (generalization) or add the new characteristics to the previous ones.
        You should write them in the instruction format.
        """.format(Correct_cases=cases_batch,characteristics=characteristics)
            characteristics=query_azure_openai_chatgpt_chat(prompt)
    return characteristics
def split_text_to_list(text):
    prompt="""Your task is to split the following characteristics and their generation instruction into a list of strings, where each string is a separate characteristic and its instruction in the list.
    The text is:
    {text}
    You should not change the content of the text.
    Please directly output these strings as a python list like this:
    ["string1","string2",...]
    """.format(text=text)
    result=query_azure_openai_chatgpt_chat(prompt)
    try:
        result=result.split("[")[1].split("]")[0]
        return eval(result)
    except:
        result=result.split('",') if '",' in result else result.split("',")
        result=[item.replace("'","").replace("[","").replace("]","").replace('"','') for item in result]
        return result
def results_analysis(failed_cases,correct_cases):
    #failed_cases,correct_cases=results_collect(model_name)
    OOA_cases,IP_cases=case_division(failed_cases)
#    pdb.set_trace()
    OOA_characteristics=Induce_OOA_cases(OOA_cases)
    IP_characteristics=Induce_IP_cases(IP_cases)
    Correct_characteristics=Induce_Correct_cases(correct_cases)
 #   pdb.set_trace()
    return split_text_to_list(IP_characteristics),split_text_to_list(OOA_characteristics),split_text_to_list(Correct_characteristics)
