from agents import AsyncOpenAI, OpenAIChatCompletionsModel
import asyncio
from agents import Agent, Runner, RunConfig
import os
from dotenv import load_dotenv
from agents import set_default_openai_client, set_tracing_disabled
import chainlit as cl
import sys

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

# Reference: https://ai.google.dev/gemini-api/docs/openai
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

set_default_openai_client(external_client)
set_tracing_disabled(True)

# Agents
spanish_agent = Agent(
    name="spanish_agent",
    instructions="You translate the user's message to Spanish",
    handoff_description="An English to Spanish translator",
    model=model
)

french_agent = Agent(
    name="french_agent",
    instructions="You translate the user's message to French",
    handoff_description="An English to French translator",
    model=model
)

italian_agent = Agent(
    name="italian_agent",
    instructions="You translate the user's message to Italian",
    handoff_description="An English to Italian translator",
    model=model
)

# Orchestrator
orchestrator_agent = Agent(
    name="orchestrator_agent",
    instructions=(
        "You are a translation agent. You use the tools given to you to translate. "
        "If asked for multiple translations, you call the relevant tools in order. "
        "You never translate on your own, you always use the provided tools."
    ),
    tools=[
        spanish_agent.as_tool(
            tool_name="translate_to_spanish",
            tool_description="Translate the user's message to Spanish",
        ),
        french_agent.as_tool(
            tool_name="translate_to_french",
            tool_description="Translate the user's message to French",
        ),
        italian_agent.as_tool(
            tool_name="translate_to_italian",
            tool_description="Translate the user's message to Italian",
        ),
    ],
    model=model
)

# ------------------ Terminal mode ------------------ #
async def main():
    print("🌍 Welcome to Translate Agent Tools")
    print("Type one of: spanish / french / italian")
    print("Or press Enter to use orchestrator (multi-language).")
    choice = input("\nWhich agent do you want? ").strip().lower()
    msg = input("Enter the text you want translated: ")

    # Pick the right agent
    if choice == "spanish":
        agent = spanish_agent
    elif choice == "french":
        agent = french_agent
    elif choice == "italian":
        agent = italian_agent
    else:
        agent = orchestrator_agent

    result = await Runner.run(agent, msg, run_config=RunConfig(workflow_name="terminal_translate"))
    print(f"\n\n✅ Final response from {agent.name}:\n{result.final_output}")

# Run only if not started by Chainlit
if __name__ == "__main__" and "chainlit" not in sys.argv[0]:
    asyncio.run(main())

# ------------------ Chainlit integration ------------------ #
@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
        content="""
# 🏢 Translate Agent Tools

Welcome! I can help translate your messages into Spanish, French, and Italian using specialized agents.  
Just tell me what you'd like to translate and into which languages!
"""
    ).send()

@cl.on_message
async def handle_message(message: cl.Message):
    user_input = message.content
    await cl.Message(content="⏳ Generating your response...").send()

    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: Runner.run_sync(
                orchestrator_agent,
                input=user_input,
                run_config=RunConfig(workflow_name="translate_workflow")
            )
        )

        await cl.Message(
            author=orchestrator_agent.name,
            content=result.final_output
        ).send()

    except Exception as e:
        await cl.Message(content=f"❌ An error occurred: {str(e)}").send()
