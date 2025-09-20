from Salesforce.sfr_embedding_code_400m_r_model import embedding_retreival
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

with open("py.txt","r") as f:
  code_text = f.read()
  
custom_separators = ['\nclass ','\ndef ', '\n\tdef ',]

python_splitter = RecursiveCharacterTextSplitter(
    separators=custom_separators,
    keep_separator=True,
    is_separator_regex=False,
    chunk_size=50,
    chunk_overlap=0
)

python_docs = python_splitter.create_documents([code_text])

query = "how the file is handling is done in the code"

top_k_retreival = embedding_retreival([query], python_docs)

# for i, top in enumerate(top_k_retreival):
#   print("-"*150)
#   print("result ->",top[i+1])
#   print("-"*150)

parser = StrOutputParser()

prompt = PromptTemplate.from_template(
"""
You are an AI coding assistant. 
A code reference or snippet is provided, and you need to:
1. Answer the Query asked to you.
2. If Answer is not known, Do not answer it or hallucinate it. "Say I dont know"
3. Strictly follow the Context of Code.

Code Snippet:
{code_reference}

query:
{query}

"""
)

retreival_runnable = RunnableLambda(lambda args : { "code_reference" : " ".join(top[i+1] for i,top in enumerate(embedding_retreival(docs = args["python_docs"], query = args["query"]))), "query" : query })

llm = OpenAI()
chain  = retreival_runnable | prompt | llm | parser

res = chain.invoke({"python_docs": python_docs, "query" : query})

print("res -> \n",res)