ffrom dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# Load environment variables (e.g., API key)
load_dotenv()

# Set up the language model
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
)

# First prompt: generate a comprehensive report
report_prompt = PromptTemplate(
    input_variables=["subject"],
    template="Provide an in-depth report about {subject}."
)

# Second prompt: condense the report into a short summary
summary_prompt = PromptTemplate(
    input_variables=["content"],
    template="Summarize the following content in exactly 5 lines:\n{content}"
)

# Generate the report based on the topic
report_input = report_prompt.invoke({"subject": "M.S Dhoni"})
detailed_report = llm.invoke(report_input)

# Generate the summary from the report
summary_input = summary_prompt.invoke({"content": detailed_report.content})
summary_output = llm.invoke(summary_input)

# Display the summary
print(summary_output.content)
