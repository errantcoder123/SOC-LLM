from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# Load environment variables
load_dotenv()

# Initialize the model
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.1,
)

# Define the structured response format
fact_schemas = [
    ResponseSchema(name="point_one", description="First interesting fact related to the subject"),
    ResponseSchema(name="point_two", description="Second interesting fact related to the subject"),
    ResponseSchema(name="point_three", description="Third interesting fact related to the subject"),
]

# Set up the parser for extracting structured facts
output_parser = StructuredOutputParser.from_response_schemas(fact_schemas)

# Create the prompt with embedded format instructions
fact_prompt = PromptTemplate(
    input_variables=["subject"],
    partial_variables={"format_instruction": output_parser.get_format_instructions()},
    template="List three unique facts about {subject}.\n{format_instruction}"
)

# Assemble the chain
pipeline = fact_prompt | llm | output_parser

# Execute the chain with the chosen topic
final_output = pipeline.invoke({"subject": "MS Dhoni"})

# Print the structured result
print(final_output)
