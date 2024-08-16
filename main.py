import os 
from  dotenv  import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from  langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_groq import ChatGroq
import pyboxen
from langchain_core.runnables import RunnablePassthrough

load_dotenv()


groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key = groq_api_key , model_name = "Llama3-8b-8192",temperature=0.5)

from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings()


db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
query = " What is Langgraph. Please provide example code?"
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
combine_docs_chain = create_stuff_documents_chain(llm,retrieval_qa_chat_prompt)
retrieval_chain = create_retrieval_chain(
    retriever= db.as_retriever() , combine_docs_chain=combine_docs_chain
)
result = retrieval_chain.invoke(input={'input':query})
boxed_output = pyboxen.boxen(str(result['answer']), padding=1, title="Query Result",color = 'red')

# Print the boxed output
print(boxed_output)


################# Retrieval Implementation with LCEL ####################


def format_focs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

template = """ 
Use the following pieces of context to answer the question at the end .
If you don't know the answer , Just say that you don't know .Don't try to make 
up an answer.
Use three sentences maximum and keep the answer as concise as possible 
Always say " Thanks for asking !" at the end of the answer 

{context}
Question:{question}

"""
custom_rag_prompt = PromptTemplate.from_template(template)

rag_chain = (
    {"context":db.as_retriever()| format_focs , "question": RunnablePassthrough()}
    | custom_rag_prompt 
    | llm
)   


res = rag_chain.invoke(query)
boxed_output = pyboxen.boxen(str(result['answer']), padding=1, title="Query Result",color = 'yellow')
print(boxed_output)
