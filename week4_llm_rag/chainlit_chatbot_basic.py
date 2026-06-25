# Basic Chainlit Chatbot
# Based on Week 4 resource: cloning ChatGPT user interface using Chainlit.
# Do not write API keys directly in this file.

import os
import chainlit as cl
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_MESSAGE = "You are a helpful assistant."

@cl.on_message
async def main(message: cl.Message):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_MESSAGE
            },
            {
                "role": "user",
                "content": message.content
            }
        ],
        temperature=0.7
    )

    answer = response.choices[0].message.content
    await cl.Message(content=answer).send()