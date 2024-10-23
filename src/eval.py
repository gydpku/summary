from openai_call import query_azure_openai_chatgpt_chat
import pdb 
import re
def evaluate_score(example, criteria):
    prompt="You are a data evaluator. Your task is to give the example score based on the given criteria. Example:{0}. Criteria:{1}. You only need to output the score number. Score number:".format(example,criteria)
    try:
        index_1=criteria.find('[')
        index_2=criteria.find(']')
        criteria=eval(criteria[index_1:index_2+1])
    except:
        criteria=criteria.replace('\n','').replace('[','').replace(']','').replace("'",'').replace('    ','') #aaa=1
    #pdb.set_trace()
    text=query_azure_openai_chatgpt_chat(prompt)
    match = re.findall(r'\b[0-5]\b', text)
    return int(match[0]) if match else None #query_azure_openai_chatgpt_chat(prompt)

def evaluate_score_reason(example, criteria):
    prompt="You are a data evaluator. Your task is to give the example score based on the given criteria. Example:{0}. Criteria:{1}. You only need to output the score and an issue analysis and revision sugestion if you do not give score 5. Score and reason:".format(example,criteria)
    try:
        index_1=criteria.find('[')
        index_2=criteria.find(']')
        criteria=eval(criteria[index_1:index_2+1])
    except:
        criteria=criteria.replace('\n','').replace('[','').replace(']','').replace("'",'').replace('    ','') #aaa=1
   
     #pdb.set_trace()
    text=query_azure_openai_chatgpt_chat(prompt)
    match = re.findall(r'\b[0-5]\b', text)
    return text #int(match[0]) if match else None
def evaluate_score_self(example, criteria):
    prompt="""You are a data evaluator responsible for assigning scores to the quality of samples on a scale of 0 to 5, with higher scores indicating better quality.You only need to output the score and the specific evaluation reason. Example:{0}""".format(example)
    text=query_azure_openai_chatgpt_chat(prompt)
    match = re.findall(r'\b[0-5]\b', text)
    return text
