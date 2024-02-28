from flask import Flask, request, jsonify
import pandas as pd

import os

from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.storage.blob import BlobServiceClient
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone
from langchain.vectorstores import Pinecone 
from langchain.document_loaders import UnstructuredPDFLoader
from sentence_transformers import SentenceTransformer
from pinecone_text.dense.sentence_transformer_encoder import SentenceTransformerEncoder 
from langchain.embeddings import SentenceTransformerEmbeddings
from flask_cors import CORS, cross_origin
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain import PromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate
from datetime import datetime
# "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"
import pandas as pd
import base64
import openai
openai.api_key = "ADD_YOUR_API_KEY_HERE"
model = SentenceTransformer('all-MiniLM-L6-v2')
os.environ["OPENAI_API_KEY"] = "ADD_YOUR_API_KEY_HERE"
os.environ["HUGGINGFACE_API_KEY"] = "ADD_YOUR_API_KEY_HERE"


conn_string = "DefaultEndpointsProtocol=https;AccountName=csg100300008bcbf08d;AccountKey=mBOuynXlYamPbMosZfaM7zJvgyLKe1fATjpYA5/nZZiJkaRJRvepRJDJCgrwNem0yVmyP0yDqc3f+ASt20Gbmg==;EndpointSuffix=core.windows.net"
cont_name = "legalcontainer"

blob_svc_client = BlobServiceClient.from_connection_string(conn_string)
blob_cont_client = blob_svc_client.get_container_client(cont_name)

Web_encoder = SentenceTransformerEncoder("sentence-transformers/all-MiniLM-L6-v2")
start_time = datetime.now().strftime("%H:%M:%S")
IndexNumber = [123,124]
PromptTester = False

acronym_dict = {"BCP": "Business Continuity Plan"}
# Example target folder
target_folder = "documents"
# List all files in target folder
files = os.listdir(target_folder)

full_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        template=
        "{prompt}",
        input_variables=["prompt"],
    )
)
chat_prompt_template = ChatPromptTemplate.from_messages([full_prompt])

llm = OpenAI(temperature=1, max_tokens=350,top_p=1)

chain = LLMChain(llm=llm, prompt=chat_prompt_template, verbose=True)

model = SentenceTransformer('all-MiniLM-L6-v2')
pinecone.init(
    api_key="508c06b2-9319-484a-a4df-6bb07761a2f9",  # find at app.pinecone.io
    environment="us-west1-gcp-free"  # next to api key in console
)

ExcelName = "SIFY-Vendor Risk Assessement Questionnaire - 2022.xlsx"

Index_Name = "langchain-chatbot"

if(Index_Name not in pinecone.list_indexes()):
    pinecone.create_index(Index_Name, dimension=384, metric="cosine")

limit = 3000
def construct_prompt(query, index, file_Name):
    # append contexts until hitting limit
    index = pinecone.Index('langchain-chatbot')
    input_em = model.encode(query).tolist()
    result = index.query(input_em, top_k=3, includeMetadata=True)
    #there is a chance of getting error and
    contexts =  result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']
    #print(contexts)
    
    prompt_start = (f"Please analyze the provided question and evidence, and verify whether the evidence is relevant to the question. If the evidence is relevant, provide a Short explanation for why it supports in the given format do the same If the evidence is not relevant. Additionally, please provide a confidence level on your answer, ranging from 0-100%. Finally, if the evidence is not in English, please indicate the language and confirm that you have understood it and provided answers in English.\n\nHere's a sample output format that the chatbot could follow:\n\nVerification Status: Evidence provided and Verified/Evidence provided but not Verified\nReason: Detailed reason from the provided pdf\nAnswer: Short Answer for the Given Query\nConfidence Level: X% (where X is a number between 0-100)\nEvidence Language: [Indicate Evidence language]\nAnswer Language: English\n\n\n Evidence:") 
    #prompt_start = (f"Understand the Given Query and context. try to get the Answer from the Context Proided(which contains the writtern policy or documention or policy or framework or process of {file_Name}), if you are able to get the Answer From the Context give me the result wheather it is verified or not with the Convensable reason and give the result in the format of Verification Status : Evidence provided and Verified / Evidence provided but not Verified\nReason: \n\n Context: \n") 
    prompt_end = (f"\n\nQuery: {query}\nAnswer:")
    for i in range(1, len(contexts)):
        if len("-".join(contexts[:i])) >= limit:
            prompt = (f"{prompt_start} {contexts} {prompt_end}")
            break
        elif i == len(contexts)-1:
            prompt = (f"{prompt_start} {contexts} {prompt_end}")
    #print(prompt)
    return prompt


def OpenAi(prompt_with_contexts):
   
    response = openai.Completion.create(
    model="gpt-3.5-turbo-instruct",
    prompt=prompt_with_contexts,
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )

    return response['choices'][0]['text']
def trans(Response_dict):
    
    translate_output_dict={}
    for each_label in Response_dict:

        
        message_text =  "translate it to english - " +Response_dict[each_label]+"""Answer should just be the translated text 
        Answer:"""
        response = openai.Completion.create(
            model="gpt-3.5-turbo-instruct",
            prompt=message_text,
            temperature=0.7,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
            )
        translate_output_dict[each_label] = response['choices'][0]['text']
    return translate_output_dict

app = Flask(__name__) 
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/make_request', methods = ["POST"]) # type: ignore
@cross_origin()
def post_request():
    data = request.get_json(force=True)  # Assuming JSON data is expected
    
    df = pd.DataFrame()
    for row in data:
        
        print(pd.DataFrame([row]))
        df=pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    
    print(df)

    
    df = df.reset_index()
    df['Answer'] = ''
    df['Evidence Path'] = ''
    df.rename(columns={"Question":"Information Security Assessment Questions"}, inplace=True)
    path_list=[]
    print(df)
    Main_df = df.copy(deep=False)
    Evidence_Answer = ""
  
    
    
    for ind, row in df.iterrows():

        query = row["Information Security Assessment Questions"]
       
        Evidence_String = row['EvidenceBinary']
        if pd.notna(Evidence_String):
            File_Availablity = False
            base64FileData  = Evidence_String
            fileData = base64.urlsafe_b64decode(base64FileData.encode('UTF-8'))
            file_name =row['RequestID']+".pdf"
            filepath =row['RequestID']+".pdf"
            #write to local file 
            with open(filepath, 'wb') as theFile:
                theFile.write(fileData)    
            
            blob_client = blob_svc_client.get_blob_client(container="legalcontainer", blob=file_name)

            print("\nUploading to Azure Storage as blob:\n\t" + file_name)

            # # Upload the created file
            # with open(file=file_name, mode="rb") as data:
            #     blob_client.upload_blob(data)

            # blob_client = blob_svc_client.get_blob_client(container=cont_name, blob=file_name)
            # # Get the blob path
            # blob_path = blob_client.url
            blob_path="Null"

            # Print the blob path
            print("Blob Path:", blob_path)
            Main_df.loc[ind, 'Evidence Path'] = blob_path
            #Do the same in pdf files also
            file_name_without_extension = os.path.splitext(filepath)[0]

        
            File_Availablity = True
            
            
            path_list.append(filepath)
            
            if ".pdf" in file_name:
                # Load the PDF file using the UnstructuredPDFLoader class
                pdf_loader = UnstructuredPDFLoader(filepath)
                pdf_document = pdf_loader.load()

                def split_docs(documents,chunk_size=1000,chunk_overlap=20):
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    docs = text_splitter.split_documents(documents)
                    return docs

                docs = split_docs(pdf_document)
                
                embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

                index_name = "langchain-chatbot"
                print(docs)
                index = Pinecone.from_documents(docs, embeddings, index_name=index_name)
        ############## we need to send list of file name*
                prompt_with_contexts = construct_prompt(query, index, file_name)
                print(prompt_with_contexts)
                Answer = OpenAi(prompt_with_contexts)
                #print(f"Answer : {Answer}")
                Evidence_Answer = Answer
                Main_df.loc[ind, 'Answer'] = Answer.strip()
   
                Answer =""

            if "pptx" in file_name:
                pass

            

       

        query = ""

        path_list = []
        
        index = pinecone.Index('langchain-chatbot')    
        try:
            delete_response = index.delete(delete_all=True)
        except:
            pass
    
    Main_df = Main_df.drop(columns=["index","EvidenceBinary"])
    os.remove(file_name)
    Main_df.rename(columns={"Information Security Assessment Questions":"Question"}, inplace=True)
    Main_df.rename(columns={"Comment (justify to your Response)":"Comment"}, inplace=True)
    Main_dict= Main_df.to_dict('records')
    return jsonify(Main_dict)











@app.route('/make_request_jpg', methods = ["POST"]) # type: ignore
@cross_origin()
def post_request_jpg():
    data = request.get_json(force=True)  # Assuming JSON data is expected
    
    df = pd.DataFrame()
    for row in data:
        
        print(pd.DataFrame([row]))
        df=pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    
    print(df)

    
    df = df.reset_index()
    df['Answer'] = ''
    df['Evidence Path'] = ''
    df.rename(columns={"Question":"Information Security Assessment Questions"}, inplace=True)
    path_list=[]
    print(df)
    Main_df = df.copy(deep=False)
    Evidence_Answer = ""
  
    
    
    for ind, row in df.iterrows():

        query = row["Information Security Assessment Questions"]
       
        Evidence_String = row['EvidenceBinary']
        if pd.notna(Evidence_String):
            File_Availablity = False
            base64FileData  = Evidence_String
            fileData = base64.urlsafe_b64decode(base64FileData.encode('UTF-8'))
            file_name =row['RequestID']+".jpg"
            filepath =row['RequestID']+".jpg"
            #write to local file 
            with open(filepath, 'wb') as theFile:
                theFile.write(fileData)    
            
            blob_client = blob_svc_client.get_blob_client(container="legalcontainer", blob=file_name)

            print("\nUploading to Azure Storage as blob:\n\t" + file_name)

            # Upload the created file
            # with open(file=file_name, mode="rb") as data:
            #     blob_client.upload_blob(data)

            # blob_client = blob_svc_client.get_blob_client(container=cont_name, blob=file_name)
            # # Get the blob path
            # blob_path = blob_client.url
            blob_path="Null"
            # Print the blob path
            print("Blob Path:", blob_path)
            Main_df.loc[ind, 'Evidence Path'] = blob_path
            #Do the same in pdf files also
            file_name_without_extension = os.path.splitext(filepath)[0]

        
            File_Availablity = True
            
            
            path_list.append(filepath)
            
            if ".jpg" in file_name:
                
                url = 'https://app.nanonets.com/api/v2/OCR/Model/1a1ba242-6a57-4095-b9b2-4e36be81a74b/LabelFile/?async=false'

                data = {'file': open(file_name, 'rb')}
                # data = {'file': open(sys.argv[1], 'rb')}

                response = requests.post(url, auth=requests.auth.HTTPBasicAuth('ADD_YOUR_API_KEY_HERE', ''), files=data)
                # response = {'message': 'Success', 'result': [{'message': 'Success', 'input': 'test file.png', 'prediction': [{'id': 'acf933be-d289-4029-b7f6-d9caa0221010', 'label': 'Address', 'xmin': 335, 'ymin': 53, 'xmax': 1665, 'ymax': 311, 'score': 0, 'ocr_text': 'ALMACONTACT S.A.S \nEntidad Contratante : Avenida Carrera . 15 # 110-45 P 6 \nBogotá , D.C. , Colombia \nsiguiente ( s ) pagina', 'type': 'field', 'status': 'correctly_predicted', 'page_no': 0, 'label_id': 'c81df03a-4cf4-4afd-9618-f70884cd54fc'}, {'id': '0eb1b8ea-ccbc-4f8e-bc18-45f59379db36', 'label': 'Certification_details', 'xmin': 232, 'ymin': 3916, 'xmax': 1828, 'ymax': 4514, 'score': 0, 'ocr_text': 'Carolina Prieto Carranza \nGerente Técnico BUREAU VERITAS \nCertification Certificado No. CO23.06187 Versión : No. 1 Fecha de Revisión : 08 mayo 2023 \nLa validez de este certificado depende de la validez del certificado principal , que expira \n31 octubre 2025 S \n4 VER RITAS \n1828 O ONAC \nACREDITADO \nISO / IEC 17021-1 : \n10 - CSG - 007', 'type': 'field', 'status': 'correctly_predicted', 'page_no': 0, 'label_id': '0ced553b-084d-4d8f-a8d5-09f51fc82929'}, {'id': '50fe7044-68e8-4d87-a681-8fae21e9066b', 'label': 'Standard', 'xmin': 52, 'ymin': 362, 'xmax': 1793, 'ymax': 536, 'score': 0, 'ocr_text': 'BVQI Colombia Ltda . certifica que el Sistema de Gestión de la organización ha sido \nauditado y se ha encontrado conforme con los requerimientos de las normas de \nSistema de Gestión que se detallan a continuación', 'type': 'field', 'status': 'correctly_predicted', 'page_no': 0, 'label_id': '370aa5bc-141a-42f9-8304-82ee0a87dfe9'}, {'id': '986340ab-5b5d-461e-8da8-f6dbb9075946', 'label': 'Certification_Cycle', 'xmin': 94, 'ymin': 1253, 'xmax': 1884, 'ymax': 1604, 'score': 0, 'ocr_text': 'Fecha de Inicio del Ciclo Original de Certificación : Fecha de Vencimiento del ciclo previo : Fecha de Auditoria de Certificación : \nFecha de Inicio del ciclo de Certificación : 08 mayo 2023 \nN / A \n27 febrero 2023 \n08 mayo 2023 Sujeto a la operación continua y satisfactoria del Sistema de Gestión de la organización , este certificado vence el : 31 octubre 2025 T', 'type': 'field', 'status': 'correctly_predicted', 'page_no': 0, 'label_id': '397b6402-4e54-4460-b08f-bbbe17be0941'}, {'id': '91519a93-2c05-47fa-a64d-3c0a8bcbf4fa', 'label': 'Declaration', 'xmin': 45, 'ymin': 870, 'xmax': 1811, 'ymax': 1183, 'score': 0, 'ocr_text': 'PRESTACIÓN DE SERVICIOS DE CONTACT CENTER Y PROCESOS ASOCIADOS A LA \nEXTERNALIZACIÓN DE SERVICIOS BPO PARA CLIENTES NACIONALES E \nINTERNACIONALES . \nDeclaración de aplicabilidad : Del 10/02/2023 . Exclusión de controles A.6.2.2 - A.14.2.7 5', 'type': 'field', 'status': 'correctly_predicted', 'page_no': 0, 'label_id': '52fc6dbd-4afa-45c4-b17e-fa62e94c81ed'}, {'id': '4fc0ea0b-1849-438c-9b9c-80927b8f4156', 'label': 'Certification_Scope', 'xmin': 0, 'ymin': 0, 'xmax': 0, 'ymax': 0, 'score': 0, 'ocr_text': 'Nombre del \nSitio \nH.O \nSITIO 1 \nSITIO 2 Dirección del Sitio \nAvenida Carrera . 15 # 110-45 \nP 6 Basik \nBogotá D.C. , Colombia \nCalle 162 # 21-91 Toberín \nBogotá D.C. , Colombia \nCarrera , 48A # 61Sur - 75 \nSabaneta , Antioquía , Colombia I Alcance del Sitio \nPRESTACIÓN DE SERVICIOS DE \nCONTACT CENTER Y PROCESOS \nASOCIADOS A LA EXTERNALIZACIÓN DE \nSERVICIOS BPO PARA CLIENTES \nNACIONALES E INTERNACIONALES . \nPRESTACIÓN DE SERVICIOS DE \nCONTACT CENTER Y PROCESOS \nASOCIADOS A LA EXTERNALIZACIÓN DE \nSERVICIOS BPO PARA CLIENTES \nNACIONALES E INTERNACIONALES . \nPRESTACIÓN DE SERVICIOS DE \nCONTACT CENTER Y PROCESOS \nASOCIADOS A LA EXTERNALIZACIÓN DE \nSERVICIOS BPO PARA CLIENTES \nNACIONALES E INTERNACIONALES .', 'type': 'field', 'status': 'correctly_predicted', 'page_no': 0, 'label_id': 'bf035ce8-321d-4793-b0a2-c44cf7e8e123'}], 'page': 0, 'request_file_id': 'f38e951c-24df-481d-8f91-0524a68ccd85', 'filepath': 'uploadedfiles/1a1ba242-6a57-4095-b9b2-4e36be81a74b/PredictionImages/348c0345-5a25-4a86-8195-c5bb9faaffda.jpeg', 'id': '3a9242b5-aeb5-11ee-953b-a61bf39d42a9', 'rotation': 0, 'file_url': 'uploadedfiles/1a1ba242-6a57-4095-b9b2-4e36be81a74b/RawPredictions/f38e951c-24df-481d-8f91-0524a68ccd85.png', 'request_metadata': '', 'processing_type': 'sync', 'size': {'width': 1991, 'height': 4721}, 'raw_ocr_api_response': {'results': None}}], 'signed_urls': {'uploadedfiles/1a1ba242-6a57-4095-b9b2-4e36be81a74b/PredictionImages/348c0345-5a25-4a86-8195-c5bb9faaffda.jpeg': {'original': 'https://nnts.imgix.net/uploadedfiles/1a1ba242-6a57-4095-b9b2-4e36be81a74b/PredictionImages/348c0345-5a25-4a86-8195-c5bb9faaffda.jpeg?expires=1704794795&or=0&s=817657bedf52e86e32aa8e4e02bec0b1', 'original_compressed': 'https://nnts.imgix.net/uploadedfiles/1a1ba242-6a57-4095-b9b2-4e36be81a74b/PredictionImages/348c0345-5a25-4a86-8195-c5bb9faaffda.jpeg?auto=compress&expires=1704794795&or=0&s=07060d17039cfdd8b59c6224228c2817', 'thumbnail': 'https://nnts.imgix.net/uploadedfiles/1a1ba242-6a57-4095-b9b2-4e36be81a74b/PredictionImages/348c0345-5a25-4a86-8195-c5bb9faaffda.jpeg?auto=compress&expires=1704794795&w=240&s=f03b81c7931a5734b873bab0bb334c28', 'acw_rotate_90': 'https://nnts.imgix.net/uploadedfiles/1a1ba242-6a57-4095-b9b2-4e36be81a74b/PredictionImages/348c0345-5a25-4a86-8195-c5bb9faaffda.jpeg?auto=compress&expires=1704794795&or=270&s=1b4c26408330870ba5336b6822d8f295', 'acw_rotate_180': 'https://nnts.imgix.net/uploadedfiles/1a1ba242-6a57-4095-b9b2-4e36be81a74b/PredictionImages/348c0345-5a25-4a86-8195-c5bb9faaffda.jpeg?auto=compress&expires=1704794795&or=180&s=edb07ce0c31db43cc20387130cade554', 'acw_rotate_270': 'https://nnts.imgix.net/uploadedfiles/1a1ba242-6a57-4095-b9b2-4e36be81a74b/PredictionImages/348c0345-5a25-4a86-8195-c5bb9faaffda.jpeg?auto=compress&expires=1704794795&or=90&s=b2dad93b5023938a3dcfabcb67a7e848', 'original_with_long_expiry': 'https://nnts.imgix.net/uploadedfiles/1a1ba242-6a57-4095-b9b2-4e36be81a74b/PredictionImages/348c0345-5a25-4a86-8195-c5bb9faaffda.jpeg?expires=1720332395&or=0&s=f772392bf7d6e31eded8b145250fc080'}, 'uploadedfiles/1a1ba242-6a57-4095-b9b2-4e36be81a74b/RawPredictions/f38e951c-24df-481d-8f91-0524a68ccd85.png': {'original': 'https://nanonets.s3.us-west-2.amazonaws.com/uploadedfiles/1a1ba242-6a57-4095-b9b2-4e36be81a74b/RawPredictions/f38e951c-24df-481d-8f91-0524a68ccd85.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA5F4WPNNTLX3QHN4W%2F20240109%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20240109T060635Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&response-cache-control=no-cache&X-Amz-Signature=0492c5822ef69d0fb38a20675d0ca4d5e5b88422c4c42140bd5a239765b5f3f8', 'original_compressed': '', 'thumbnail': '', 'acw_rotate_90': '', 'acw_rotate_180': '', 'acw_rotate_270': '', 'original_with_long_expiry': ''}}}
                response = response.json()
                data = response['result'][0]['prediction']
                Response_dict = {}
                for item in data:
                    Response_dict[item['label']] = item['ocr_text']

                translate_output_dict=trans(Response_dict)

                          
            

                def split_docs(translate_output_dict,chunk_size=1000,chunk_overlap=20):
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                   
                    string_list=' '.join(list(translate_output_dict.values()))
                    print([string_list])
                    docs = text_splitter.create_documents([string_list])
                    return docs

                docs = split_docs(translate_output_dict)
                print(docs)
                embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

                index_name = "langchain-chatbot"
                
                index = Pinecone.from_documents(docs, embeddings, index_name=index_name)
        ############## we need to send list of file name*
                prompt_with_contexts = construct_prompt(query, index, file_name)
                print(prompt_with_contexts)
                Answer = OpenAi(prompt_with_contexts)
                #print(f"Answer : {Answer}")
                Evidence_Answer = Answer
                Main_df.loc[ind, 'Answer'] = Answer.strip()
   
                Answer =""

        
       

        query = ""

        path_list = []
        
        index = pinecone.Index('langchain-chatbot')    
        try:
            delete_response = index.delete(delete_all=True)
        except:
            pass
    
    Main_df = Main_df.drop(columns=["index","EvidenceBinary"])
    os.remove(file_name)
    Main_df.rename(columns={"Information Security Assessment Questions":"Question"}, inplace=True)
    Main_df.rename(columns={"Comment (justify to your Response)":"Comment"}, inplace=True)
    Main_dict= Main_df.to_dict('records')
    return jsonify(Main_dict)




if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
