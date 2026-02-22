from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


loader = PyPDFLoader("data/circ4001bc.pdf")
documents = loader.load()

# Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100
)

texts = text_splitter.split_documents(documents)

#  Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Criar index FAISS
vectorstore = FAISS.from_documents(
    texts,
    embeddings
)


vectorstore.save_local("storage/faiss_index")

