import langchain, langchain_community, langchain_openai
import os
from langchain import hub
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY')
os.environ["HUGGINGFACE_API_KEY"] = os.getenv("HUGGINGFACE_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
from langchain_groq import ChatGroq
llm_groq = ChatGroq(model="llama3-70b-8192", temperature=0)
from langchain_openai import ChatOpenAI



question = "How much total debt is held by Tata Motors? Compared to Reliance Industries, which company can come out of debt quicker?"



llm_4o = ChatOpenAI(
    model="gpt-4o",
    # model="gpt-3.5-turbo-16k",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

llm_3 = ChatOpenAI(
    # model="gpt-4o",
    model="gpt-3.5-turbo-16k",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

import yfinance as yf
def get_stock_price(symbol: str) -> float:
    try :
        stock = yf.Ticker(symbol)
        price = str(stock.info)
        print(price)
        return price
    except:
        print("Please rephrase the company name according to the stock market.\n")
        quit()

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful assistant that returns only the company ticker name from user query and nothing else.
            The user might not mention the whole company name (like Ltd or Pvt), so make sure you understand the context and provide the correct ticker symbol.
            If the company is Indian, add .NS to the ticker symbol.
            If user query does not contain a valid company, return 'none'.
            If there are multiple companies in the query, return each one separated by \n.""",
        ),
        ("human", "{input}"),
    ]
)

from langchain_core.output_parsers import StrOutputParser
chain = prompt | llm_groq | StrOutputParser()

indexes = chain.invoke({"input":question}).split()

data_of_company = ""
for i in indexes:
    print(i)
    data_of_company+=get_stock_price(i)

## Data infestion

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

## Data infestion

from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.schema.document import Document
from langchain_community.vectorstores import Chroma
def get_text_chunks_langchain(data_of_company):
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200,separators="\n")
    docs = [Document(page_content=x) for x in text_splitter.split_text(data_of_company)]
    return docs

docs = get_text_chunks_langchain(data_of_company)
# docs = get_text_chunks_langchain(result)
documents = filter_complex_metadata(docs)

## Vector embeddings and Vector Store
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings

ollama_emb = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma()
try:
    db._collection.delete(db.get()['ids'])
    db = Chroma.from_documents(documents, ollama_emb)
except:
    db = Chroma.from_documents(documents, ollama_emb)

import datetime

now = datetime.datetime.now()
dates = now.strftime("%Y-%m-%d %H:%M:%S")
dates

result = db.similarity_search(question)

prompt1 = ChatPromptTemplate.from_template(
    """Answer questions related to Stock Exchange and the Stock Market, by understanding the context and the abbreviations mentioned in the questions and context.
    You are forbidden from answering any question unrelated to Stock Exchange and the Stock Market.
    The context you are given is data scraped from the web about the user question related to stocks.
    Understand the stock market jargons mentioned in the context and answer the user query. Do not mention that you have a context given in the response.

Context: {context}

Question: {question}


If date and time is required, consider the date and time as {dates} in %Y-%m-%d %H:%M:%S format.
"""
)

chain = (
    prompt1
    | llm_4o
    | StrOutputParser()
)

print(chain.invoke({"question": question, "context": result, "dates": dates}))