from dotenv import load_dotenv
from pydantic import BaseModel 
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool
from tools import definition_tool
import subprocess

###from langchain_gemini import ChatGemini

load_dotenv()

class ConjecturegenerationResponse(BaseModel):
    topic: str  
    summary: str              # Topic of the conjecture (e.g., "group theory", "number theory")
    conjecture: str
    source: list[str]    # List of sources or references that inspired the conjecture
    explanation: str  
    tools_used : list[str]          # Source of the conjecture (e.g., "mathlib", "user input")

llm = ChatOpenAI(model="gpt-3.5-turbo")

###llm2 = ChatGemini(model="gemini-1.5-pro", api_key="your_gemini_api_key")

parser = PydanticOutputParser(pydantic_object=ConjecturegenerationResponse)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an advanced mathematical reasoning assistant. Your task is to generate insightful and original mathematical conjectures based on a given topic.
            You must understand the core ideas in the topic.
            Avoid restating known theorems.
            Output a plausible, original conjecture in the style of mathematical research.
            Clearly explain the reasoning behind the conjecture.
            Warp this output in this formal and provide no other text\n{format_instructions}
            """,
        ),
("placeholder", "{chat_history}"),
("human", "{query}"),
("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, definition_tool]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
query = input("what can I help you with?")
raw_response = agent_executor.invoke({"query": query})
print(raw_response)

try:
    structured_response = parser.parse(raw_response.get("output")[0]["text"])
    print("Structured Response:", structured_response)
except Exception as e:
    print("Error parsing response:", e, "Raw Response:", raw_response)