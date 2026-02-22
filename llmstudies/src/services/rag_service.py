class RAGService:
    def __init__(self,vector_store, llm_service):
        self.vector_store = vector_store
        self.llm = llm_service

    def query(self, question:str):
        docs = self.vector_store.similarity_search(question, k=3)

        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
        Responda a pergunta usando o contexto abaixo.
        Se não souber, diga que não encontrou na Circular 4001.

        Contexto:
        {context}

        Pergunta:
        {question}
        """

        return self.llm.generate(prompt)