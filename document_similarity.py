from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


load_dotenv()

embed_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-exp-03-07", 
    dimensions=300
)

cricket_bios = [
    "Virat Kohli is known for his aggressive cricketing style and has captained the Indian team.",
    "MS Dhoni led India to multiple trophies and is famous for staying calm under pressure.",
    "Sachin Tendulkar is regarded as a cricket legend with an unmatched list of records.",
    "Rohit Sharma has several double centuries in ODIs and plays with fluent strokeplay.",
    "Jasprit Bumrah is a pace bowler recognized for his deadly yorkers and unique action."
]

user_input = "who is bumrah"

bio_vectors = embed_model.embed_documents(cricket_bios)
input_vector = embed_model.embed_query(user_input)


similarities = cosine_similarity([input_vector], bio_vectors)[0]


top_match_idx = int(np.argmax(similarities))
top_score = similarities[top_match_idx]

print("Query:", user_input)
print("Top Match:", cricket_bios[top_match_idx])
print("Cosine Similarity Score:", top_score)
