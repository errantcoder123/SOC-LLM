from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

vectorizer = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=32
)

sentence = "The capital city of India is Delhi."

embedding_vector = vectorizer.embed_query(sentence)

print(str(embedding_vector))
