from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


language_model = OpenAI(model="gpt-3.5-turbo-instruct")

question = "Name the capital city of India."


response = language_model.invoke(question)

print(response)
