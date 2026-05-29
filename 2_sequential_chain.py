from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import os

load_dotenv()

os.environ["LANGCHAIN_PROJECT"] = "sequential llm app"

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

api_key = os.getenv("GEMINI_API_KEY")
model1 = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", api_key=api_key, temperature=0.7)
model2 = ChatGoogleGenerativeAI(model="gemini-3.5-flash", api_key=api_key, temperature=0.5)

parser = StrOutputParser()

chain = prompt1 | model1 | parser | prompt2 | model2 | parser

config = {
    "run_name": "sequential_chain",
    "tags": ["report", "summary"],
    "metadata": {"model1_temperature": 0.7, "model2_temperature": 0.5, "model1_name": "gemini-3.1-flash-lite", "model2_name": "gemini-3.5-flash"}
}

result = chain.invoke({'topic': 'Unemployment in India'}, config=config)

print(result)
