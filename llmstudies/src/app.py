from fastapi import FastAPI
from pydantic import BaseModel

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

app = FastAPI()

class UserInput(BaseModel):
    input: str

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    "storage/faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

llm = Ollama(model="tinyllama")


prompt = ChatPromptTemplate.from_template("""
Responda usando apenas o contexto abaixo.
Se não encontrar a resposta na Circular 4001, diga que não encontrou.

Contexto:
{context}

Pergunta:
{input}
""")

document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)


@app.post("/query")
def query(user_input: UserInput):
    result = retrieval_chain.invoke({"input": user_input.input})
    return {"answer": result["answer"]}