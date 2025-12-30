
from typing import Optional, List, Union, Tuple
from openai import OpenAI
from pathlib import Path
import os
from dotenv import load_dotenv
import base64
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

def initialize_qwen_multimodal_api():
    llm = ChatOpenAI(
        model="qwen-vl-plus",
        api_key=os.getenv("QWEN_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    return llm