

from fastapi import FastAPI
from pydantic import BaseModel

class UserInput(BaseModel):
    input: str

app = FastAPI()

@app.post('/query')
def get_user_input(user_input: UserInput):
    return UserInput(input=user_input.input)

