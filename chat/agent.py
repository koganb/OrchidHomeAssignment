from __future__ import annotations

from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from chat.tools import create_chat_tools

from dotenv import load_dotenv



def run_react_agent(question: str, chroma_path: str, networkx_path: str, collection_name: str) -> str:
    llm = _get_llm()

    tools = create_chat_tools(
        chroma_path=chroma_path,
        networkx_path=networkx_path,
        collection_name=collection_name,
    )
    prompt = _react_prompt(PromptTemplate)
    agent = create_react_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)
    result = executor.invoke({"input": question})
    return result["output"]


def _get_llm():
    load_dotenv()
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


def _react_prompt(prompt_template_class):
    return prompt_template_class.from_template(
        """Answer only repository code and execution-flow explanation questions using the available tools.
Do not answer general knowledge, writing, translation, personal advice, or non-repository questions.

You have access to the following tools:

{tools}

Use this format:

Question: the input question
Thought: think about what context is needed
Action: the action to take, one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... this Thought/Action/Action Input/Observation can repeat
Thought: I now know the final answer
Final Answer: the final answer to the original question

Question: {input}
Thought:{agent_scratchpad}"""
    )

