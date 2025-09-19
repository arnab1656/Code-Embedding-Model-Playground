from Salesforce.sfr_embedding_code_400m_r_model import embedding_retreival
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

try:
 with open("py.txt","r") as f:
    code_text = f.read()
except Exception as e:
    print("Cannot acces the File")

PYTHON_CODE = [code_text]

# custom_separators = ['\nclass ', '\ndef ', '\n\tdef ', '\n\n', '\n', ' ', '']

# custom_separators = ['\nclass ', '\ndef ', '\n\tdef ', '\n\n']

custom_separators = ['\nclass ','\ndef ', '\n\tdef ',]

python_splitter = RecursiveCharacterTextSplitter(
    separators=custom_separators,
    keep_separator=True,
    is_separator_regex=False,
    chunk_size=50,
    chunk_overlap=0
)

python_docs = python_splitter.create_documents(PYTHON_CODE)

print("-"*150)

for i, docs in enumerate(python_docs):
  print("At index ---> ",i+1)
  print("docs is ----> \n\n",docs.page_content,"\n\n")

  print("-"*150)

python_docs_text = [p.page_content for p in python_docs]

query = ["make a snake letter in python"]

best_passage, top_k_retreival = embedding_retreival(query, python_docs_text)

print("best_passage --> \n" ,best_passage)

for i, top in enumerate(top_k_retreival):
   print("-"*150)
   print("top ->",top[i+1])


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

query = "make a snake letter in python"

llm = OpenAI()
chain = prompt | llm

res = chain.invoke({"code_reference": code_docs, "query" : query})

print("res -> \n",res)