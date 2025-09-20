from Salesforce.sfr_embedding_code_400m_r_model import embedding_retreival
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

with open("py.txt","r") as f:
  code_text = f.read()
  
# print(code_text)

custom_separators = ['\nclass ','\ndef ', '\n\tdef ',]

python_splitter = RecursiveCharacterTextSplitter(
    separators=custom_separators,
    keep_separator=True,
    is_separator_regex=False,
    chunk_size=50,
    chunk_overlap=0
)

python_docs = python_splitter.create_documents([code_text])

query = "File handling in python"

top_k_retreival = embedding_retreival([query], python_docs)

for i, top in enumerate(top_k_retreival):
  print("-"*150)
  print("result ->",top[i+1])
  print("-"*150)

  
prompt = PromptTemplate.from_template(
"""
You are an AI coding assistant. 
The user will provide a code reference or snippet, and you need to:
1. Explain what the code does in simple terms.
2. Suggest improvements or best practices if applicable.
3. Provide example usage if relevant.

Code Snippet:
{code_reference}

query:
{query}

"""
)

code_docs = " ".join( top[i+1] for i,top in enumerate(top_k_retreival))

llm = OpenAI()
chain = prompt | llm

res = chain.invoke({"code_reference": code_docs, "query" : query})

print("res -> \n",res)