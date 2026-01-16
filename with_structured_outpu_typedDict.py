from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# ✅ Define the function to describe output structure
def parse_review() -> dict:
    """
    Returns:
        summary: A concise summary of the review
        sentiment: The overall sentiment (positive, negative, or neutral)
    """
    pass

structured_model = model.with_structured_output(parse_review)

result = structured_model.invoke(
    "I really like the display quality and battery life on this device. However, the fingerprint sensor is slow and sometimes unresponsive. The speaker quality could also be better. Overall, it's a decent choice for the price."
)

print(result)

