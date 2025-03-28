from langchain.agents import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
from pydantic import BaseModel, Field
import datetime
import random
import json
import re

from operate_user_bio import generate_topics_by_bio


class StoryInput(BaseModel):
    user_full_name: str = Field(description="full name of user")
    topic: str = Field(description="topic of story to write")

@tool(args_schema=StoryInput)
def story_generation(user_full_name: str, topic: str) -> str:
    """if you need to write, create or generate story in Instagram."""
    return f"generate story text spoken from the author for {user_full_name} on topic {topic}, don't use hashtags and smiles. Put it between <story> and </story> tags. Generate a short (30 words) image background description for this story. Put it between <bgr> and </bgr> tags"
    
class ReelsInput(BaseModel):
    user_full_name: str = Field(description="full name of user")
    topic: str = Field(description="topic of reels to write")
    duration: int = Field(description="desirable length of reel in seconds")

@tool(args_schema=ReelsInput)
def reels_generation(user_full_name: str, topic: str, duration: int) -> str:
    """if you need to write, create or generate reel(s) in Instagram."""
    return f"generate reel message for {user_full_name} on topic {topic} and duration {duration} seconds"

tools = [story_generation, reels_generation]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are very powerful and creative assistant ready to produce involving and thought-provoking content. 
            You have some useful tools. If they give an answer, return it directly without adding some words""",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

from dotenv import load_dotenv
import httpx
import os
load_dotenv(override=True)

OPEN_AI_API_KEY = os.getenv("OPENAI_API_KEY")
PROXY = os.getenv("OPENAI_PROXY_SOCKS")
OPENAI_RESPONSE_TIMEOUT_SECONDS = 25

if PROXY:
    _http_client = httpx.Client(
        proxies=PROXY, timeout=httpx.Timeout(OPENAI_RESPONSE_TIMEOUT_SECONDS)
    )
else:
    _http_client = httpx.Client(
        timeout=httpx.Timeout(OPENAI_RESPONSE_TIMEOUT_SECONDS)
    )

llm = ChatOpenAI(api_key=OPEN_AI_API_KEY, model="gpt-3.5-turbo", temperature=0.9, http_client=_http_client)
llm_with_tools = llm.bind_tools(tools)

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def execute_story_request(prompt: str):
    res = list(agent_executor.stream({"input": prompt}))

    return res

def generate_story_prompts(user_full_name: str):
    week_end_date = datetime.datetime.now().strftime("%Y-%m-%d")
    topics = generate_topics_by_bio(user_full_name = user_full_name, week_end_date = week_end_date)
    generated_topic = topics[random.randint(0,len(topics)-1)]

    prompt = f"Hi! I am {user_full_name}. Please help me with a story for Instagram. The topic of the story: {generated_topic}?"
    result = execute_story_request(prompt = prompt)

    story = []
    bgr = []
    for i in range(len(result)):
        if 'output' in result[i].keys():
            story = re.findall("<story>(.*?)</story>", result[i]['output']) 
            bgr = re.findall("<bgr>(.*?)</bgr>", result[i]['output'])
    
    story = story[0] if story else ""
    bgr = bgr[0] if bgr else ""
    return story, bgr


if __name__ == "__main__":
    user_full_name = "Olivia Silverleaves" 

    story, bgr = generate_story_prompts(user_full_name = user_full_name)
    print(f"Story: {story}")
    print(f"Background: {bgr}")
