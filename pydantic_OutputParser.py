from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Initialize the Groq model
model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.1,
)

# Define a Pydantic model for structured output
class Person(BaseModel):
    name: str = Field(description='Name of the person')
    age: int = Field(description='Age of the person')
    city: str = Field(description='City the person belongs to')

# Initialize the parser with the Pydantic model
parser = PydanticOutputParser(pydantic_object=Person)

# Create a prompt template
prompt = PromptTemplate(
    template="Generate the name, age, and city of a fictional person from {place}.\n{format_instructions}",
    input_variables=["place"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# Create the chain
chain = prompt | model | parser

# Invoke the chain with the desired input
result = chain.invoke({"place": "Indore"})

# Print the result
print(result)
