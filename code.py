from flask import Flask, request, jsonify, render_template
app = Flask(__name__, template_folder='Templates')# Set a secret key for the Flask application
app.secret_key = 'my_secret_key'
from werkzeug.datastructures import FileStorage
import re
from datetime import datetime
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import time
import json
import fitz
import ast
import os
from openai import AzureOpenAI
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import (
                            StuffDocumentsChain,
                            LLMChain,
                            ReduceDocumentsChain,
                            MapReduceDocumentsChain,
                            load_summarize_chain,
                        )
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter, Document
from langchain.document_loaders import PyPDFLoader
import pdfplumber


client = AzureOpenAI(
  azure_endpoint = "azure endpoint", 
  api_key="api key",  
  api_version="2023-08-01-preview"
)

def get_completion(prompt):
    response = client.chat.completions.create(
        model='AcademyGenAI',
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0)
    response = response.choices[0].message.content
    return response

def form_reading_pdf(filename):
    print("inside form recognizer")
    endpoint = "form recognizer endpoint"
    key = "form recognizer key"
    # st = time.time()
    client = DocumentAnalysisClient(endpoint, AzureKeyCredential(key))
 
    # Use the custom model to extract information from the document
    with open(filename, "rb") as f:
        poller = client.begin_analyze_document("prebuilt-layout", f.read())
        result = poller.result()
 
    output = {}
    page_number = 1  # Initialize page number
    for page in result.pages:
        text = f"Page number: {page_number}, "
        for line in page.lines:
            text += line.content + "\n"
        output['Page number ' + str(page_number)] = text.strip()
        page_number += 1  # Increment page number
 
    # et = time.time()
    # print("Time taken by reading pdf:", et - st)
 
    return json.dumps(output)

def form_convert_dict_to_list_of_dicts(input_string):
    input_dict = ast.literal_eval(input_string)
    list_of_dicts = []
    for key, value in input_dict.items():
        list_of_dicts.append({key: value})
    return list_of_dicts

def convert_dict_to_list_of_dicts(input_dict):
    # input_dict = ast.literal_eval(input_string)
    list_of_dicts = []
    for key, value in input_dict.items():
        list_of_dicts.append({key: value})
    return list_of_dicts

def reading_pdf(filename):
    print("inside reading pdf func",filename)
    doc = fitz.open(filename)
    page = doc[0]
    if len(page.get_text())>100:
        print("********pdf is editable*********",len(page.get_text()))
        final_dict = {}
        for i in range(len(doc)):
            page = doc[i]  # The third page is at index 2 
            page_content = page.get_text()
            page_number_key = f"Page number {i+1}"
            final_dict[page_number_key] = page_content
            final_dict_1 = convert_dict_to_list_of_dicts(final_dict)
        return final_dict_1

    else:
        print("**************pdf is not editable*************")
        final_dict = form_reading_pdf(filename)
        final_dict_1 = form_convert_dict_to_list_of_dicts(final_dict)
        return final_dict_1

model1 = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

def get_embeddings(text):
    print("inside embeddings")
    embeddings = model1.encode(text)
    return embeddings

def find_closely_related_chunks(query, embeddings, text, threshold=0.0, top_k=20):
    similarity_scores = cosine_similarity([query], embeddings)[0]
    above_threshold_indices = np.where(similarity_scores > threshold)[0]
    top_k_indices = above_threshold_indices[np.argsort(similarity_scores[above_threshold_indices])[::-1][:top_k]]
    print("Top 3 similarity scores:",format(top_k))
    for idx in top_k_indices:
        print("Index {}: {}".format(idx, similarity_scores[idx]))
        
    closely_related_chunks = [{key: text[idx][key] for key in text[idx]} for idx in top_k_indices]
    return closely_related_chunks    


def get_response(user_query, chunk):
    print("user_query------------------------",user_query)
    output_format = """[{"<user_query>":"<Identified answer>"}]"""
    prompt = f"""
    You are a legal contract validation specialist. You will be provided a two inputs: a 'user query' and a 'text'.\n
    Your task is to extract details based on the user query within the provided text. Provide complete and more detailed responses with maximum information.\n
    Response should be answered properly to the user query.\n
    Display the response strickly in a output format {output_format}.
    user_query: {user_query}
    text: {chunk} 
    """
    response = get_completion(prompt)
    return response


def repharse_response(text):
    
    output_format = """[{"<user_query>":"<Identified answer>"}]"""
    prompt = f"""
    You are a rephrase assistance for legal contract details. You will be provide a list of dictionary.\n
    Dictionary contains a 'key' respresenting the 'user query' and a corresponding 'value' representing the 'answer'.\n
    Answer to the user query *only* by rephrasing whole knowledge given to you.\n
    If the user asks a question, begin by answering "Yes" or "No" before providing an explanation, based on the user query.\n 
    You should always be Polite & Professional, and always respond in Standard American English.\n
    Display the response strickly in the specified output format {output_format}.

    text: {text} 
    """
    response = get_completion(prompt)
    return response

#--------Summary--------------------------------------------
def extract_images_and_text(filename):
    with pdfplumber.open(filename) as pdf:
        text = ''
        for page in pdf.pages:
            page_text = page.extract_text()
            text += page_text + '\n'
    return text  

def analyze_layout(filename):
                    # sample form document
    endpoint = "form recognizer endpoint"
    key = "form recognizer key"
    document_analysis_client = DocumentAnalysisClient(
        endpoint=endpoint, credential=AzureKeyCredential(key)
    )
    with open((filename), "rb") as f:
        poller = document_analysis_client.begin_analyze_document(
        "prebuilt-layout", document=f
    )
    image_result = poller.result()
    return image_result.content

def extract_text_from_pdf(filename):
    res = extract_images_and_text(filename)
    if len(res)>100:
        print("pdf is editable")
        return res
    else:
        print("pdf is not editable")
        res = analyze_layout(filename)
        return res

@app.route('/', methods=["POST", "GET"])
def home():
    return render_template("index.html")
 
@app.route('/process_data', methods=['POST'])
def process_file():
    if request.method == 'POST':
        query = request.form['query']
        print('query--------------',query,type(query))
        if 'file' not in request.files:
            return render_template("index.html", result='invalid file')
        file = request.files['file']
        if file.filename == '':
            return render_template("index.html", result='invalid file')

        if file and file.filename.endswith(('.pdf')):
            print('file is present')
            st1 = time.time()
            file.save(file.filename)  # Save the uploaded file temporarily
            extracted_text = reading_pdf(file.filename)
            os.remove(file.filename)
            et1 = time.time()
            print('extracted_text',et1-st1)
            st = time.time()
            pdf_embedding = get_embeddings(extracted_text)
            et = time.time()
            print('pdf_embedding',et-st)
            st2 = time.time()
            query_embedding = get_embeddings(query)
            et2 = time.time()
            print('query_embedding',et2-st2)
            st3 = time.time()
            top_chunks = find_closely_related_chunks(query_embedding, pdf_embedding, extracted_text)
#             print('top_chunks-------------',top_chunks)
            et3 = time.time()
            print('top_chunks',et3-st3)
            st4 = time.time()
            gpt_call = get_response(query, top_chunks)
            gpt_call = eval(gpt_call)
            print('gpt_call---------------',gpt_call)
            et4 = time.time()
            print('gpt_call',et4-st4)
            repharse = repharse_response(gpt_call)
            print('*********repharse-----------------',repharse,type(repharse))
            repharse = eval(repharse)
            print('repharse----------',repharse)
            print('repharse----------',type(repharse))    
            
            values = [list(d.values())[0] for d in repharse]
            print('values---------------------',values,type(values))
            combined = ' '.join(values)
            print('combined-------------------',combined,type(combined))
            
            final = [{query: combined}]
            print('final----------------------',final)
            output = json.dumps(final)
            print('output---------------------',output)
            
            return output
        else:
            return render_template("index.html", result='invalid file')
        
@app.route('/summary', methods=['POST'])
def summary():
    print('inside summary')
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template("index.html", result='invalid file')
        file = request.files['file']
        if file.filename == '':
            return render_template("index.html", result='invalid file')

        if file and file.filename.endswith(('.pdf')):
            print('file is present')
            st1 = time.time()
            file.save(file.filename)
            res = extract_text_from_pdf(file.filename)  
            os.remove(file.filename)
            
        #     print('res---------',res)
            text_splitter = CharacterTextSplitter()

            # Assuming clean_text is a string
            with open('temp.txt','w',encoding='utf-8') as f:
                f.write(res)

            # Now, read the text file and pass it to the splitter
            with open('temp.txt','r',encoding='utf-8') as f:
                sampletxt = f.read()

            text_splitter = CharacterTextSplitter(
                separator=" ",
                chunk_size=28000,
                chunk_overlap=0,
                length_function=len,
                is_separator_regex=False,
            )
            texts = text_splitter.create_documents([sampletxt])
        #     print('texts-----------------',texts)

            llm = AzureChatOpenAI(
                        openai_api_key="open ai key",
                        azure_endpoint="endpoint",
                        openai_api_version="2023-08-01-preview",
                        model_name="gpt-35-turbo-16k",
                        deployment_name="deployement name"
                        )

            target_len=6000

            Dateprompt_template = """Write a summary of the financial aspect of the contract between the parties, covering the following points:

                                    - The contract start date and end date, which indicate the period of time during which the contract is valid and enforceable.
                                    - The effective date, which is the date when the contract becomes legally binding and the obligations of the parties begin.
                                    - The deal duration, if it is available, which specifies the length of time for which the contract is intended to last or the conditions for its termination.
                                    - The related data from the contract that support or illustrate the key points, such as the names of the parties, the subject matter of the contract, the rights and duties of the parties, the payment terms, the dispute resolution mechanism, and any other relevant clauses or provisions.


                                    The summary should be concise, clear, and accurate, and should not exceed 4000 words. 
                                    The summary should also use the same language and terminology as the contract, and should avoid any interpretation or opinion that is not based on the contract itself.
                                    "{text}"
                                    CONCISE SUMMARY:"""


            DatePROMPT = PromptTemplate(template=Dateprompt_template, input_variables=["text"])

            Daterefine_template = (
                                "Your job is to produce a final summary\n"
                                "We have provided an existing summary up to a certain point: {existing_answer}\n"
                                "We have the opportunity to refine the existing summary"
                                "(only if needed) with some more context below.\n"
                                "------------\n"
                                "{text}\n"
                                "------------\n"
                                f"Given the original summary Highlight Start date,End date,Effective Date,Deal Duration and original summary in English within {target_len} words"
                            )

            Daterefine_prompt = PromptTemplate(
                                input_variables=["existing_answer", "text"],
                                template=Daterefine_template)

            Datechain = load_summarize_chain(
                llm=llm,
                chain_type="refine",
                #return_intermediate_steps=True,
                question_prompt=DatePROMPT,
                refine_prompt=Daterefine_prompt,
                input_key="input_documents",
                #verbose=True,

            )

            Dateresult = Datechain({"input_documents": texts}, return_only_outputs=True)
            print('Dateresult-------------------',Dateresult)
            return Dateresult
        else:
            return render_template("index.html", result='invalid file')

        
        
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False) 
