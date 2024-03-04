from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Qdrant
from qdrant_client.http.models import Distance, VectorParams
from langchain_community.vectorstores import qdrant
from langchain_openai import OpenAIEmbeddings
from qdrant_client.http.models import Distance, VectorParams
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from qdrant_client.http import models
from qdrant_client import QdrantClient
from langchain.chains import LLMChain
from dotenv import load_dotenv
import qdrant_client
from openai import OpenAI
import getpass
import os


load_dotenv()

url = os.getenv('Qdrant_URL')
qrdant_api_key=os.getenv('Qdrant_API_KEY')
qdrant_client = QdrantClient(url=url, api_key= qrdant_api_key)


# try: 
#     qdrant_client.recreate_collection(
#     collection_name="teesne-collection",
#     vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
# ) 
# except Exception as e:
#     print(f'Failed to create the collection: {e}')


api_key = os.getenv('OpenAI_API_Key')
embeddings = OpenAIEmbeddings(api_key=api_key)
vector_store = Qdrant(
    client=qdrant_client, collection_name="teesne-collection", 
    embeddings=embeddings,
)


def get_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator = ">>>",
        chunk_size=1000, 
        chunk_overlap= 200,
        length_function=len
        )
    chunks = text_splitter.split_text(text)
    return chunks


with open('para.txt') as f:
    raw_text = f.read()
texts=get_chunks(raw_text)
vector_store.add_texts(texts)



from langchain_openai import OpenAI
qa = RetrievalQA.from_chain_type( llm=OpenAI(), chain_type = "stuff",
                                 retriever= vector_store.as_retriever()
                                )          

                      


# querry = "I'm going on my friends birthday. Suggest me any gift for a birthday event"
# response = qa.invoke(querry)
# print(response)

# querry2 = "I'm going on a date with my girlfriend, Suggest me a gift that would be romantic?"
# response2 = qa.invoke(querry2)
# print(response2)