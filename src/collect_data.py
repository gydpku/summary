import fasttext
import os
#from haystack.document_stores import InMemoryDocumentStore, FAISSDocumentStore
#from haystack.nodes import BM25Retriever, EmbeddingRetriever
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
import re
import torch
import subprocess
import datasets
from datasets import load_dataset,load_from_disk,concatenate_datasets
from collections import Counter
from openai_call import query_azure_openai_chatgpt_chat
from ken_ppl import KenlmModel
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
#import nltk
import sentencepiece
import pdb
import numpy as np
import random
import time
from datasets import disable_caching
disable_caching()
divide_num=4
train_num=1000
def extract_list(text):
    index1=text.find('[')
    index2=text.find(']')
    return eval(text[index1:index2+1])
def prompt_synonyms(text):
    return 'You can generate synonyms about this domain:{0}. Each synonym must have the same meaning as the domain. Your output must only be a list of all synonyms like ["xxx","xxx"].'.format(text) 
def prompt_keywords(text):
    return 'You can generate 50 diverse subcategories and their relevant weight (0~1) about this domain: {0}. We prefer to word rather than phrase. Your output must be a list of all domain keywords and their relevant weight (float number) like ["(word,weight)","(xxx,xxx)"]'.format(text) 
#'You can generate 50 diverse and strong-correlated keywords about this domain:{0}. We prefer to word rather than phrase. Your output must be a list of all domain keywords like ["xxx","xxx","xxx"]'.format(text)
def train_kenlm(input_file,kenlm_path,output_file, order=5):
    command = f"{kenlm_path} -o {order} < {input_file} > {output_file}"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print("Error during KenLM training:")
        print(stderr.decode('utf-8'))
    else:
        print("KenLM training completed successfully.")
        print(stdout.decode('utf-8'))
def prompt_domain_find_wikipassage(domain_name,passage_num=5000):
    domain_syns=query_azure_openai_chatgpt_chat(prompt_synonyms(domain_name)) 
    domain_syns=extract_list(domain_syns) #query_prompt_synonyms(text)
    domain_keys_weights=query_azure_openai_chatgpt_chat(prompt_keywords(domain_name))
    domain_keys_weights=extract_list(domain_keys_weights)
    domain_weights={}
    domain_keys=[]
    for item in domain_keys_weights:
        item=str(item)
        domain_keys.append(item.split(',')[0].replace('(','').lower())
        try:
            domain_weights[item.split(',')[0].replace('(','').lower()]=float(item.split(',')[1].replace(')',''))
        except:
            pdb.set_trace()

    for item in domain_syns:
        domain_keys.append(item.lower())
        domain_weights[item.lower()]=0 #1000 #float('inf')
    domain_keys.append(domain_name.lower())
    domain_weights[domain_name.lower()]=0 #float('inf')
    print('Generating keywords for your domain:'+ str(domain_name)+ str(domain_keys))
    wiki = load_dataset("wikimedia/wikipedia", "20231101.en",cache_dir='/dccstor/obsidian_llm/yiduo') 
    pdb.set_trace()
    documents=[]
    for data in wiki['train']:
        documents.append(Document(page_content=data['text'],metadata={"name": data['title']}))

    retriever = BM25Retriever.from_documents(
    documents)
    pdb.set_trace()
    def word_num_count(examples):
        results = []
        for text in examples['text']:
            words = text.lower().split()
            total_words = len(words)
            results.append(total_words)
        return {'num': results}
    def indicator(num):
        return 0 if num==0 else 1
    def word_count(examples):
        results = []
        for text in examples['text']:
            words = text.lower().split()
            total_words = len(words)
            keyword_counts = Counter(word for word in words if word in domain_keys)
            appearance_rate = sum((indicator(keyword_counts[keyword])*domain_weights[keyword]) / total_words for keyword in domain_keys)
            if keyword_counts[domain_name]>0:
                appearance_rate+=(indicator(keyword_counts[domain_name])*10)/ total_words
            if any(keyword_counts[syn]>0 for syn in domain_syns):
                appearance_rate+=sum([(indicator(keyword_counts[keyword])*10)/ total_words for keyword in domain_syns]) #1000 #float('inf') 
            results.append(appearance_rate)
        return {'app_rate': results}
    wiki_counted = wiki['train'].map(
    word_count, 
    batched=True, 
    batch_size=1000,  # Adjust batch size as needed
    num_proc=32, 
    desc="Calculating frequency on dataset",
load_from_cache_file=False,
)
    wiki_counted_num = wiki_counted.map(
    word_num_count, 
    batched=True, 
    batch_size=1000,  # Adjust batch size as needed
    num_proc=32, 
    desc="Calculating word number on dataset",
load_from_cache_file=False,
)
    wiki_counted_num=wiki_counted_num.sort('num') #,reverse=True)
    wiki_counted_num.save_to_disk('/dccstor/obsidian_llm/yiduo/wiki_counted_num_ori')
    wiki_counted_num=load_from_disk('/dccstor/obsidian_llm/yiduo/wiki_counted_num_ori')
#    pdb.set_trace()    
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    '''
    high_texts=[]
    for id in range(divide_num):
        print(id,'\n')
        start_index = (len(wiki_counted_num) // divide_num) * id
        end_index = (len(wiki_counted_num) // divide_num) * (id+1)
        high_texts.extend([example['text'] for example in wiki_counted_num.select(range(start_index,end_index)).sort('app_rate', reverse=True).select(range(min(200//divide_num,end_index-start_index)))])
    import pdb
#    pdb.set_trace()
    model_path = 'Mihaiii/gte-micro-v3' #'Alibaba-NLP/gte-large-en-v1.5' #Mihaiii/gte-micro-v3
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path) #tokenizer = AutoTokenizer.from_pretrained(model_path)
    mean_vector=None
    for text in high_texts:
        encoded_input=tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        if mean_vector is None:
            mean_vector=np.array(mean_pooling(model(**encoded_input),encoded_input['attention_mask']).tolist()[0])
        else:
            mean_vector+=np.array(mean_pooling(model(**encoded_input),encoded_input['attention_mask']).tolist()[0])
    mean_vector/=len(high_texts)
    ''' #model = AutoModel.from_pretrained(model_path, trust_remote_code=True,cache_dir='/dccstor/obsidian_llm/yiduo')
    dataset=wiki['train']
    
    def get_coses(examples):
        results=[]
    #    tokenizer = AutoTokenizer.from_pretrained(model_path)
     #   model = AutoModel.from_pretrained(model_path, trust_remote_code=True,cache_dir='/dccstor/obsidian_llm/yiduo')
        for text in examples['text']:
            #tokenizer = AutoTokenizer.from_pretrained(model_path)
            #model = AutoModel.from_pretrained(model_path, trust_remote_code=True,cache_dir='/dccstor/obsidian_llm/yiduo')
            encoded_input=tokenizer(text, padding=True, truncation=True, return_tensors='pt')
            vector=np.array(mean_pooling(model(**encoded_input),encoded_input['attention_mask']).tolist()[0])
            dot_product = np.dot(mean_vector, vector)

# Step 2: Compute the norm of each vector
            norm_a = np.linalg.norm(mean_vector)
            norm_b = np.linalg.norm(vector)

# Step 3: Compute the cosine similarity
            cosine_similarity = dot_product / (norm_a * norm_b)
            results.append(cosine_similarity) #mean_pooling(model(**encoded_input),encoded_input['attention_mask']).tolist()[0])
        return {"cos": results}
    '''
    import pdb
    pdb.set_trace()
    dataset=dataset.map(get_embedding)
    batch_dict = tokenizer(high_texts[:10], max_length=8192, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**batch_dict)
    domain_embeddings = outputs.last_hidden_state[:, 0]
    domain_embeddings = F.normalize(domain_embeddings, p=2, dim=1).mean(0).unsqueeze(0)
    def get_simi(examples): 
        results=[] 
        #import fasttext
        texts=[content for content in examples['text']]
        batch_dict = tokenizer(texts, max_length=8192, padding=True, truncation=True, return_tensors='pt')
        outputs = model(**batch_dict)
        embeddings = outputs.last_hidden_state[:, 0]
        embeddings = F.normalize(embeddings, p=2, dim=1) #.mean(0).unsqueeze(0)  #model = fasttext.load_model("domain_model.bin")
        results=(domain_embeddings @ embeddings.T).tolist()
        results=results[0] if len(results[0])!=1 else results
        return {'simi':results} 
    pdb.set_trace()
    wiki_simi = wiki['train'].map(get_simi,num_proc=16,batched=True,batch_size=10,desc="calculating smi on dataset",load_from_cache_file=False)    #
    '''
    
#    wiki_counted_num=wiki_counted_num.sort('num')
 #   wiki_counted_num.save_to_disk('/dccstor/obsidian_llm/yiduo/wiki_counted_num_ori')
    
    #wiki_counted_num_0=wiki_counted_num.filter(lambda x: x['app_rate'] > 10)
    #wiki_counted_num_0.save_to_disk('/dccstor/obsidian_llm/yiduo/wiki_counted_num_0') 
    #wiki_counted_num_0=load_from_disk('/dccstor/obsidian_llm/yiduo/wiki_counted_num_0')
    #if len(wiki_counted_num_0)>passage_num:
    #    return wiki_counted_num_0.sort('app_rate',reverse=True).select(range(passage_num))
#    pdb.set_trace()
    wiki_counted_num_1=wiki_counted_num.filter(lambda x: x['app_rate'] > 0 ) #and x['app_rate'] <1000)
    wiki_counted_num_1.save_to_disk('/dccstor/obsidian_llm/yiduo/wiki_counted_num_1') 
    wiki_counted_num_1=load_from_disk('/dccstor/obsidian_llm/yiduo/wiki_counted_num_1') 
    wiki_counted_num_2=wiki_counted_num.filter(lambda x: x['app_rate']==0) #wiki_counted_num=wiki_counted_num.filter(lambda x: x['app_rate'] > 0) #wiki_counted_num=wiki_counted_num.sort('num')
    wiki_counted_num_2.save_to_disk('/dccstor/obsidian_llm/yiduo/wiki_counted_num_2') #    pdb.set_trace()
    wiki_counted_num_2=load_from_disk('/dccstor/obsidian_llm/yiduo/wiki_counted_num_2') #   wiki_counted_num=wiki_counted_num.filter(lambda x: x['app_rate'] > 0)
    wiki_counted_num_2=wiki_counted_num_2.shuffle() #   pdb.set_trace()
    pdb.set_trace() #positive_samples_num_sorted=positive_samples.sort('num')
    texts=[]
    labels=[]
    if len(wiki_counted_num_1)>passage_num:
        key_texts=[]
        for id in range(divide_num):
            print(id,'\n')
            start_index = (len(wiki_counted_num_1) // divide_num) * id
            end_index = (len(wiki_counted_num_1) // divide_num) * (id+1)
            key_texts.extend([example['text'] for example in wiki_counted_num_1.select(range(start_index,end_index)).sort('app_rate', reverse=True).select(range(min((passage_num)//(divide_num),len(wiki_counted_num_1) // divide_num)))]) #texts.extend([example['text'] for example in wiki_counted_num_0.sort('app_rate', reverse=True).select(range(min(train_num,len(wiki_counted_num_0))))])
        return key_texts
    pdb.set_trace() #labels=['High' for i in range(len(texts))]
    medium=[]
    for id in range(divide_num):
        print(id,'\n')
        start_index = (len(wiki_counted_num_1) // divide_num) * id
        end_index = (len(wiki_counted_num_1) // divide_num) * (id+1)
        #pdb.set_trace()
        medium.extend([example['text'] for example in wiki_counted_num_1.select(range(start_index,end_index)).sort('app_rate', reverse=True).select(range(min((train_num)//(divide_num),len(wiki_counted_num_1) // divide_num)))])
    pdb.set_trace() #        texts.extend([example['text'] for example in positive_samples_num_sorted.select(range(start_index,start_index+min(50,len(positive_samples) // 4)))]) #end_index = (len(wiki_counted_num_sorted) // 4) * (id + 1)
    
    texts.extend(medium)
    labels.extend(['Positive' for i in range(len(medium))])
    #texts.extend([example['text'] for example in wiki_counted_num_2.select(range(min(train_num//2,len(wiki_counted_num_2))))])
    #labels.extend(['negative' for i in range(min(train_num//2,len(wiki_counted_num_2)))])
    add_nega=[example['text'] for example in wiki_counted_num_2.select(range(min(train_num*5,len(wiki_counted_num_2))))]
    texts.extend(add_nega)
    labels.extend(['Negative' for i in range(len(add_nega))])
#    pdb.set_trace()
    index_random=[i for i in range(len(texts))] 
    random.shuffle(index_random) #[i for i in range(len(texts))])
 #   val_texts = [texts[i] for i in index_random][-100:]  #texts=texts[index_rando    val_labels = [labels[i] for i in index_random][-100:] #pdb.set_trace()
    train_texts = [texts[i] for i in index_random] #[:-100]  #texts=texts[index_random]
    train_labels = [labels[i] for i in index_random] #[:-100]  #labels=labels[index_random]
    #add_nega=[example['text'] for example in wiki_counted_num_2.select(range(min(train_num*1000,len(wiki_counted_num_2))))]
    #train_texts.extend(add_nega)
    #train_labels.extend(['negative' for i in range(len(add_nega))])
    def generate_train_file(texts, labels, output_file):
        with open(output_file, 'w') as f:
            for text, label in zip(texts, labels):
                f.write(f"__label__{label} {text}\n")

# Generate the train.txt file
    generate_train_file(train_texts, train_labels, "wiki_keywords_train.txt")
#    generate_train_file(val_texts, val_labels, "wiki_keywords_val.txt")
    # Select the subset of the dataset
        #temp_dataset = wiki_counted_num_sorted[start_index:end_index]
    #temp_dataset=wiki_counted_num_sorted.select(range((len(wiki_counted_num_sorted)//4)*id,(len(wiki_counted_num_sorted)//4)*(id+1)))
        #temp_dataset_sorted = sorted(temp_dataset, key=lambda x: x['app_rate'], reverse=True) #temp_dataset_sorted=temp_dataset.sort('app_rate', reverse=True)
        #model = fasttext.train_supervised(
    #input="wiki_keywords_train.txt",
    import fasttext
#    pdb.set_trace()
    model = fasttext.train_supervised(input="wiki_keywords_train.txt",epoch=100,lr=1.0,wordNgrams=5) #autotuneValidationFile="wiki_keywords_val.txt") 
    model.save_model("domain_model.bin") #autotuneValidationFile="wiki_keywords_val.txt"top_samples = temp_dataset_sorted[:min(10000, len(temp_dataset_sorted))] #top_samples = temp_dataset_sorted.select(range(min(10000, len(temp_dataset_sorted))))
    #model = fasttext.load_model("domain_model.bin")    #high_app_rate = [sample for sample in top_samples if sample['app_rate'] > 0][:50] #high_app_rate = top_samples.filter(lambda x: x['app_rate'] > 0).select(range(50))
        #texts.extend([example['text'] for example in high_app_rate])
    #pdb.set_trace()
    
    '''
    texts=medium
    sp = sentencepiece.SentencePieceProcessor()
    sp.load('en.sp.model')
    print('Find {0} relevant passages based on key words'.format(len(texts))) #top_1000 = wiki_counted_sorted.select(range(min(1000, len(wiki_counted_sorted)))) #wiki_counted_sorted = wiki_counted.sort('app_rate', reverse=True)
  #  pdb.set_trace(180)
    encode_sentences=[]
    for text in texts:
        for sentence in nltk.sent_tokenize(text.replace('\n','')):
            encode_sentences.append(" ".join(sp.encode_as_pieces(sentence.lower())))
    #pdb.set_trace()
    random.shuffle(encode_sentences)
    #pdb.set_trace()
    with open('wiki_keywords.txt', 'w') as file:
        for line in encode_sentences:
            file.write(line+'\n')
            #for sentence in nltk.sent_tokenize(text.replace('\n','')):
#model = fasttext.train_supervised(input="train_data.txt", epoch=25, lr=1.0, wordNgrams=2)            file.write(line+'\n')     
    arpa_file='domain.arpa'
    file_name='wiki_keywords.txt'
    kenlm_path='/dccstor/obsidian_llm/yiduo/kenlm/bin/lmplz'
    train_kenlm(file_name,kenlm_path, arpa_file, order=3)
    model = KenlmModel.from_pretrained('wiki','domain')
#    print('Train the KenLM model')
     '''
    def replace_newlines(text: str) -> str:
        return re.sub("\n+", " ", text)
    def get_ppl(examples):
        content = examples['text']

    # Define the conditions that return infinite perplexity
        invalid_conditions = [
        content in ['.', './n', '.\n', '\n.', ', .\n'],
        '_________________' in content,
        '__________' in content,
        '*************************' in content,
        '.........' in content,
        '~~~~~~~~~~~~~~' in content,
        '$$$$$$$$$$$$$$$$$$$$$$$$' in content,
        '\n\n\n' in content,
        '....' in content,
        len(content) < 50]
    
    # Check if any condition for infinite perplexity is met
        if any(invalid_conditions):
            return {'ppl': float('inf')}
    
    # Otherwise, calculate and return the perplexity using ngram_model
        return {'ppl': model.get_perplexity(content)}
    def get_ppls(examples):
        results=[]
        for content in examples['text']:
            invalid_conditions = [
            content in ['.', './n', '.\n', '\n.', ', .\n'],
        '_________________' in content,
        '__________' in content,
        '*************************' in content,
        '.........' in content,
        '~~~~~~~~~~~~~~' in content,
        '$$$$$$$$$$$$$$$$$$$$$$$$' in content,
        '\n\n\n' in content,
        '....' in content,
        len(content) < 50]
    
    # Check if any condition for infinite perplexity is met
            if any(invalid_conditions):
                results.append(float('inf'))
            else:
                results.append(model.get_perplexity(content))
        return {'ppl':results}
    
    def get_predicts(examples): 
        results=[] 
        import fasttext
        model = fasttext.load_model("domain_model.bin")
        for content in examples['text']:
            if len(content)<50:
                results.append(0.0)
                continue 
            pred = model.predict(replace_newlines(content)) #   wiki_num_sorted=load_from_disk('/dccstor/obsidian_llm/yiduo/wiki_counted_num_ori') 
            if 'Positive' in pred[0][0]:
                pred_posi_prob=pred[1][0]
            #elif 'High' in pred[0][0]:
            #    pred_posi_prob=pred[1][0]*10       
            else:
                pred_posi_prob=1-pred[1][0]
            #pred_posi_prob=pred[1][0] if 'Medium' in pred[0][0] else 1-pred[1][0]
            results.append(pred_posi_prob)
        return {'posi':results} 
    wiki_num_sorted=load_from_disk('/dccstor/obsidian_llm/yiduo/wiki_counted_num_ori')
        
    ppl_texts=[] 
    for id in range(divide_num):
        print(id,'\n')
        start_index = (len(wiki_num_sorted) // divide_num) * id
        end_index = (len(wiki_num_sorted) // divide_num) * (id+1)
        #ppl_texts.append(wiki_num_sorted.select(range(start_index,end_index)).map(get_predicts,num_proc=1,batched=True,batch_size=1000,desc="calculating posi",load_from_cache_file=False).sort('posi',reverse=True).select(range(min((passage_num)//divide_num,end_index-start_index))))  #pdb.set_trace()
        ppl_texts.extend([example['text'] for example in wiki_num_sorted.select(range(start_index,end_index)).map(get_predicts,num_proc=16,batched=True,batch_size=1000,desc="calculating posi on dataset",load_from_cache_file=False).sort('posi',reverse=True).select(range(min(passage_num*3//divide_num,end_index-start_index)))]) #    pdb.set_trace() #time.sleep(5)
    print('tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    list_1=[tokenizer(text, padding=True, truncation=True, return_tensors='pt') for text in ppl_texts]
    print('tokenizer,1')
    #pdb.set_trace()
    list_2=[np.array(mean_pooling(model(**encoded_input),encoded_input['attention_mask']).tolist()[0]) for encoded_input in list_1]
    print('tokenizer,2')
    list_3=[np.dot(mean_vector, vector) for vector in list_2]
    print('tokenizer,3')
    list_4=[list_3[id]/(np.linalg.norm(mean_vector)*np.linalg.norm(vector)) for id,vector in enumerate(list_2)]
    print('tokenizer,4')
    paired_list = list(zip(ppl_texts, list_4))
    sorted_paired_list = sorted(paired_list, key=lambda x: x[1], reverse=True)
    sorted_texts = [text for text, _ in sorted_paired_list]
    ppl_texts=sorted_texts[:passage_num] #pdb.set_trace()
    '''
    if len(wiki_counted_num_0)+len(wiki_counted_num_1)>passage_num:
        wiki_ppl=concatenate_datasets([wiki_counted_num_0,wiki_counted_num_1]).map(get_predicts,num_proc=16,batched=True,batch_size=1000,desc="calculating posi on dataset",load_from_cache_file=False)
        wiki_ppl_sorted = wiki_ppl.sort('posi',reverse=True)
        return wiki_ppl_sorted.select(range(passage_num))
    else:
    '''
    '''
    wiki_ppl = wiki['train'].map(get_predicts,num_proc=16,batched=True,batch_size=1000,desc="calculating ppl on dataset",load_from_cache_file=False) #wiki_ppl = wiki['train'].map(get_ppl,num_proc=8,desc="calculating ppl on dataset",)    
    wiki_ppl_sorted = wiki_ppl.sort('posi',reverse=True)
    print('Find relvant passages based on posi')
    return wiki_ppl_sorted.select(range(passage_num))
    '''
    
    return ppl_texts
