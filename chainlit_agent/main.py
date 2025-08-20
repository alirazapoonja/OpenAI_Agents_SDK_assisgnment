import asyncio
from agents import Agent , Runner , AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from dotenv import load_dotenv
import os
import chainlit as cl

# Load environment variables from .env file
load_dotenv()

#api key for Gemini API
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set in the environment variables.")


# Initialize the external client for Gemini API
extarnal_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Initialize the OpenAIChatCompletionsModel with the external client    
model = OpenAIChatCompletionsModel(
    openai_client=extarnal_client,
    model="gemini-2.0-flash",    
)

# Create a RunConfig with the model and external client
config = RunConfig(
    model=model,
    model_provider=extarnal_client,
    tracing_disabled=True
)

# Create Writer Agent
tracking_agent = Agent(
    name="✍️ Tracking Agent",
    instructions="""
        You are a helpful Tracking assistant. You can write track ship cargo,
        track shipments, and provide updates on delivery status.
        Use the Gemini API to generate responses. 
        
    """
)
# Welcome message with big heading
@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
        content="""
# 🏢 Tracking Cargo Agent

Welcome! I can help you track your cargo shipments and provide updates on their status.
Just type what you want me to write, 

🟢 Just type your container number!

"""
    ).send()

# Respond to user input
@cl.on_message
async def handle_message(message: cl.Message):
    user_input = message.content

    await cl.Message(content="⏳ Generating your response...").send()

    try:
        # Run synchronous Runner inside async using thread executor
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: Runner.run_sync(
                tracking_agent,
                input=user_input,
                run_config=config
            )
        )

        # Send result to Chainlit UI
        await cl.Message(
            author=tracking_agent.name,
            content=result.final_output
        ).send()

    except Exception as e:
        await cl.Message(content=f"❌ Error: {str(e)}").send()