from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


load_dotenv()

json_parser = JsonOutputParser()

groq_model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.1,
)


prompt_template = PromptTemplate(
    template="Please provide a fictional Indian name, age, and city in JSON format.\n{format_instruction}",
    input_variables=[],
    partial_variables={'format_instruction': json_parser.get_format_instructions()}
)


chain = prompt_template | groq_model | json_parser


output = chain.invoke({})

# Display the result
print(output)
