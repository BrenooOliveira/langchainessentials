


from httpx import request


class LLMService:

    def __init__(self,model='tinyllama') -> None:
        self.model = model
        self.url = "http://localhost:11434/api/generate"
    
    def generate(self, prompt:str):
        response  = request.post(
            self.url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
        )   

        return response.json()['response']