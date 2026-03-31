from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader

# 1. LOAD PDF
doc_path = "data/shahzeb_swe_2026.pdf"
model = "llama3.2"
if doc_path.endswith(".pdf"):
    loader = PyPDFLoader(doc_path)
else:
    loader = OnlinePDFLoader(doc_path)
data = loader.load()

print("done loading...")
# print(data[0].page_content)

# 2.Extract PDF and chunk
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(data)
# print(f"Split {len(chunks)} chunks from {len(data)} pages")

# 3. Embed chunks and store in Chroma
# using  nomic-embed-text ollama model

vector_db= Chroma.from_documents(
    documents=chunks,
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
    collection_name="simple-rag"
)

print("done embedding and storing...")

# 4. Retrival 

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_ollama import ChatOllama

from langchain_core.runnables import RunnablePassthrough
from langchain_classic.retrievers.multi_query import MultiQueryRetriever


llm= ChatOllama(model=model)

QUERY_PROMPT = PromptTemplate(
input_variables=["question"],
template="""
You are an AI language Model assistant . Your task is to generate 
5 diffrent version of user question to retrive relevant documents
from a vector database. By generating multiple versions of user questions
goal is to help user overcome limitation of distance based similarity search.
Provide these alternative questions sepearted by newlines.
Original Question: {question}
"""
)

retriever = MultiQueryRetriever.from_llm(
    retriever=vector_db.as_retriever(),
    llm=llm,
    prompt=QUERY_PROMPT
)

# 5. RAG CHAIN

template="""
    Use the following context to answer the question.
    If the answer is not in the context, say "I don't know".
    Context: {context}
    Question: {question}
    """

prompt= ChatPromptTemplate.from_template(template)


rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# 6. ASK QUESTION
question = "What does Shahzeb do?"
answer = rag_chain.invoke(question)
print(answer)