from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load API key and environment variables
load_dotenv()

# Initialize the LLM
groq_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
)

# First prompt: generate a full report
detailed_prompt = PromptTemplate(
    input_variables=["theme"],
    template="Create an in-depth write-up about {theme}."
)

# Second prompt: summarize the report into 5 lines
summary_prompt = PromptTemplate(
    input_variables=["body"],
    template="Condense the following text into a 5-line summary:\n{body}"
)

# Output parser
text_parser = StrOutputParser()

# Define the chain manually
first_step = detailed_prompt | groq_llm | text_parser
second_step = summary_prompt | groq_llm | text_parser

# Run the chain
detailed_text = first_step.invoke({"theme": "Virat Kohli"})
summary_result = second_step.invoke({"body": detailed_text})

# Output final summary
print(summary_result)
