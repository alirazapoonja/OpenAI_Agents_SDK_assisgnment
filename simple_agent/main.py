from agents import Agent , Runner , AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from dotenv import load_dotenv
import os

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set in the environment variables.")


#Reference: https://ai.google.dev/gemini-api/docs/openai
extarnal_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
    
model = OpenAIChatCompletionsModel(
    openai_client=extarnal_client,
    model="gemini-2.0-flash",    
)

config = RunConfig(
    model=model,
    model_provider=extarnal_client,
    tracing_disabled=True
)

agent = Agent(
    name="Simple Agent",
    instructions="A simple agent that can answer questions.",
)

result = Runner.run_sync(
    agent,
    "what is the capital of pakistan?",  
    run_config=config,
)

print(f"Result: {result.final_output}")