from dotenv import load_dotenv

load_dotenv()

from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langchain_tavily import TavilySearch

llm = ChatOllama(model="llama3.2:3b", temperature=0)
tools = [TavilySearch()]
agent = create_react_agent(model=llm, tools=tools)


def main():
    print("=== My search agent using Tavily ===")
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Search for 3 job postings for an ai engineer using langchain in Luxembourg on linkedin and list their details and links!"
                )
            ]
        }
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
