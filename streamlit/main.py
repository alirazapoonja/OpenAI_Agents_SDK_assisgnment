import os
import asyncio
import streamlit as st
from dotenv import load_dotenv

from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Streamlit page setup
st.set_page_config(page_title="Tracking Cargo Information", layout="centered", page_icon="🤖")
st.markdown("""
    <style>
    body {
        background-color: ##6d3f6b;
        color: white;
    }
    .stTextInput>div>div>input {
        background-color: ##0000ff;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 Tracking Cargo Information")
st.subheader("How can I Help, tracking your cargo Information!")

# Check for missing API key
if not gemini_api_key:
    st.error("❌ GEMINI_API_KEY is missing. Please set it in a `.env` file.")
    st.stop()

# Set up Gemini as OpenAI-compatible client
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    openai_client=external_client,
    model="gemini-2.0-flash",
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)
# Create Writer Agent
tracking_agent = Agent(
    name="✍️ Tracking Agent",
    instructions="""
        You are a helpful Tracking assistant. You can write track ship cargo,
        track shipments, best shipping line, best cargo rates, and provide updates on delivery status.
        You will receive a container number from the user, and you will provide the latest status of the cargo.
        If you don't know the answer, just say "I don't know".
             
    """
)
# Streamlit input
query = st.text_input("Type your container number here:", placeholder="e.g., ABCD1234567")

if st.button("Click for Answer") and query:
    with st.spinner("Searching Caro tracking..."):
        try:
            result = asyncio.run(Runner.run(tracking_agent, query, run_config=config))
            st.success("✅ Answer:")
            st.markdown(f"**{result.final_output}**")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")