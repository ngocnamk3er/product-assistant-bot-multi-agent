# =============================================================================
# agents/google_adk/agent.py
# =============================================================================
# 🎯 Purpose:
# This file defines an AI agent called EventIntroductionAgent.
# It uses Google's ADK, Gemini model, and can search a Milvus knowledge base.
# =============================================================================

# -----------------------------------------------------------------------------
# 📦 Built-in & External Library Imports
# -----------------------------------------------------------------------------
import os
from datetime import datetime  # Used to get the current system time

# 🧠 Gemini-based AI agent provided by Google's ADK
from google.adk.agents.llm_agent import LlmAgent

# 📚 ADK services for session, memory, and file-like "artifacts"
from google.adk.sessions import InMemorySessionService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.artifacts import InMemoryArtifactService
from google.adk.tools.function_tool import FunctionTool

# 🏃 The "Runner" connects the agent, session, memory, and files into a complete system
from google.adk.runners import Runner

# 🧾 Gemini-compatible types for formatting input/output messages
from google.genai import types

# import google.generativeai as genai # For embedding
from google import genai

# 🔐 Load environment variables (like API keys) from a `.env` file
from dotenv import load_dotenv

# Milvus client
from pymilvus import connections, utility, Collection
from google.genai.types import EmbedContentConfig

load_dotenv()  # Load variables like GOOGLE_API_KEY into the system

# -----------------------------------------------------------------------------
# ⚙️ Configuration for Milvus and Embedding
# -----------------------------------------------------------------------------
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# if not GOOGLE_API_KEY:
#     raise ValueError("GOOGLE_API_KEY not found in environment variables.")
# genai.configure(api_key=GOOGLE_API_KEY)

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
# MILVUS_USER = os.getenv("MILVUS_USER") # Uncomment if your Milvus needs auth
# MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD") # Uncomment if your Milvus needs auth
MILVUS_COLLECTION_NAME = (
    "event_knowledge_base"  # <<< THAY THẾ BẰNG TÊN COLLECTION CỦA BẠN
)
EMBEDDING_MODEL_NAME = "embedding-001"  # Or "models/text-embedding-004" etc.
VECTOR_FIELD_NAME = (
    "embedding"  # <<< THAY THẾ BẰNG TÊN TRƯỜNG VECTOR TRONG COLLECTION CỦA BẠN
)
TEXT_CONTENT_FIELD_NAME = (
    "text_content"  # <<< THAY THẾ BẰNG TÊN TRƯỜNG CHỨA NỘI DUNG TEXT
)
SOURCE_FIELD_NAME = "source_document"  # <<< TÙY CHỌN: TÊN TRƯỜNG CHỨA NGUỒN GỐC
TOP_K_RESULTS = 3

# -----------------------------------------------------------------------------
# 🛠️ Helper Functions for Milvus Tool
# -----------------------------------------------------------------------------


def get_text_embedding(text: str, task_type="RETRIEVAL_QUERY") -> list[float]:
    """Generates embedding for the given text using Google's API."""
    try:
        result = genai.Client().models.embed_content(
            model=EMBEDDING_MODEL_NAME,
            contents=text,
            config=EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",  # Optional
                output_dimensionality=3072,  # Optional
                title="Driver's License",  # Optional
            ),
        )
        # print(f"Generated embedding for text: {result.embeddings[0].values}")  # Debugging
        return result.embeddings[0].values
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return []


def search_milvus_knowledge_base(search_query: str) -> str:
    """
    Searches a Milvus knowledge base for information relevant to the search_query.

    Args:
        search_query (str): The user's query to search for in the knowledge base.

    Returns:
        str: A string containing the search results, or a message if no results are found or an error occurs.
    """
    try:
        # 1. Connect to Milvus (consider managing connection lifecycle better for production)
        print(f"Connecting to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")
        connections.connect(
            alias="default",
            host=MILVUS_HOST,
            port=MILVUS_PORT,
            # user=MILVUS_USER, # Uncomment for auth
            # password=MILVUS_PASSWORD, # Uncomment for auth
        )
        print("Successfully connected to Milvus.")

        # 2. Check if collection exists
        if not utility.has_collection(MILVUS_COLLECTION_NAME, using="default"):
            return f"Milvus collection '{MILVUS_COLLECTION_NAME}' not found."

        collection = Collection(MILVUS_COLLECTION_NAME, using="default")
        collection.load()  # Load collection into memory for searching
        print(
            f"Collection '{MILVUS_COLLECTION_NAME}' loaded. Num entities: {collection.num_entities}"
        )

        # 3. Generate embedding for the search query
        print(f"Generating embedding for query: '{search_query}'")
        query_embedding = get_text_embedding(search_query)
        if not query_embedding:
            return "Failed to generate embedding for the search query."

        # 4. Perform search
        search_params = {
            "metric_type": "L2",  # Or "IP" (Inner Product) depending on your data/preference
            "params": {
                "nprobe": 10
            },  # Example search param, adjust based on your index type
        }

        output_fields = [TEXT_CONTENT_FIELD_NAME]
        if SOURCE_FIELD_NAME:  # Only include source if it's configured
            # Ensure SOURCE_FIELD_NAME is part of your collection schema if you use it
            if SOURCE_FIELD_NAME in [field.name for field in collection.schema.fields]:
                output_fields.append(SOURCE_FIELD_NAME)
            else:
                print(
                    f"Warning: SOURCE_FIELD_NAME '{SOURCE_FIELD_NAME}' not found in collection schema. It will not be retrieved."
                )

        print(
            f"Searching Milvus with vector, top_k={TOP_K_RESULTS}, output_fields={output_fields}"
        )
        results = collection.search(
            data=[query_embedding],
            anns_field=VECTOR_FIELD_NAME,
            param=search_params,
            limit=TOP_K_RESULTS,
            expr=None,  # Optional: filter expression
            output_fields=output_fields,
            consistency_level="Strong",  # Or "Bounded"
        )

        collection.release()  # Release collection from memory
        print("Search complete. Collection released.")

        # 5. Format results
        if not results or not results[0]:
            return "No relevant information found in the knowledge base for your query."

        formatted_results = "Found the following information from the knowledge base:\n"
        for i, hit in enumerate(results[0]):
            content = hit.entity.get(TEXT_CONTENT_FIELD_NAME, "N/A")
            formatted_results += f"\nResult {i+1} (Score: {hit.distance:.4f}):\n"
            formatted_results += f"Content: {content}\n"
            if SOURCE_FIELD_NAME and SOURCE_FIELD_NAME in output_fields:
                source = hit.entity.get(SOURCE_FIELD_NAME, "N/A")
                formatted_results += f"Source: {source}\n"

        return formatted_results

    except Exception as e:
        print(f"Error during Milvus search: {e}")
        import traceback

        traceback.print_exc()
        return f"An error occurred while searching the knowledge base: {str(e)}"
    finally:
        try:
            connections.disconnect("default")
            print("Disconnected from Milvus.")
        except Exception as e:
            print(f"Error disconnecting from Milvus: {e}")


# -----------------------------------------------------------------------------
# 🕒 EventIntroductionAgent: Your AI agent that tells the time and searches KB
# -----------------------------------------------------------------------------


class EventIntroductionAgent:
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(self):
        self._agent = self._build_agent()
        self._user_id = "event_introduction_user"
        self._runner = Runner(
            app_name=self._agent.name,
            agent=self._agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )

    def _build_agent(self) -> LlmAgent:
        def calculate_number_of_tokens(text: str) -> int:
            return len(text.split())

        def calculate_length_of_input(
            text: str,
        ) -> int:  # Fixed typo: "leng" to "length"
            return len(text)

        # Create FunctionTool for Milvus search
        # The ADK will infer the schema from the function's signature and docstring
        milvus_kb_tool = FunctionTool(search_milvus_knowledge_base)

        return LlmAgent(
            model="gemini-1.5-flash-latest",
            name="event_introduction_and_kb_agent",
            description="Responds to event information, searches a knowledge base, and can announce sales.",
            instruction=(
                "You are a helpful AI assistant for event information. "
                "If the user asks a specific question about an event, product, or any detail that might be in a knowledge base, "
                "first use the 'search_milvus_knowledge_base' tool to find relevant information. "
                "Use the information returned by the tool to answer the user's question. "
                "If the tool returns 'No relevant information found...' or an error, inform the user you couldn't find specific details. "
                "If the user asks a general greeting or a question not suitable for a KB search, "
                "you can reply that there is a big sale coming up at the end of the year. "
                "Do not make up information if it's not found by the tool."
            ),
            tools=[
                FunctionTool(calculate_number_of_tokens),
                FunctionTool(calculate_length_of_input),
                FunctionTool(search_milvus_knowledge_base),  # Add the new Milvus tool
            ],
        )

    async def invoke(self, query: str, session_id: str) -> str:
        session = await self._runner.session_service.get_session(
            app_name=self._agent.name, user_id=self._user_id, session_id=session_id
        )
        if session is None:
            session = await self._runner.session_service.create_session(
                app_name=self._agent.name,
                user_id=self._user_id,
                session_id=session_id,
                state={},
            )

        content = types.Content(role="user", parts=[types.Part.from_text(text=query)])

        events = list(
            self._runner.run(
                user_id=self._user_id, session_id=session.id, new_message=content
            )
        )

        if not events or not events[-1].content.parts:
            return ""

        # 6) Otherwise, join all text parts of the final event and return
        return "\n".join(p.text for p in events[-1].content.parts if p.text)



    async def stream(self, query: str, session_id: str):
        # This stream method is still the simple time-telling one,
        # not integrated with the LLM or Milvus for true streaming of LLM responses.
        yield {
            "is_task_complete": True,
            "content": f"The current time is: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        }


# --- Example Usage (for testing locally) ---
async def main():
    agent = EventIntroductionAgent()
    session_id = "test_session_milvus_001"

    print("Agent initialized. Type 'quit' to exit.")

    # Example: Pre-populate Milvus (you'd do this separately in a real setup)
    # Ensure your Milvus server is running and the collection `event_knowledge_base`
    # is created with appropriate fields (e.g., id (primary, auto-id), embedding (float_vector, dim=768),
    # text_content (varchar), source_document (varchar)).
    # For this example, we'll assume it's populated.

    # Test Milvus search directly (optional)
    # print("\n--- Testing Milvus search function directly ---")
    # test_search_result = search_milvus_knowledge_base("Tell me about the summer festival")
    # print(test_search_result)
    # print("---------------------------------------------\n")

    while True:
        user_query = input("You: ")
        if user_query.lower() == "quit":
            break

        if not user_query.strip():
            continue

        print("Agent thinking...")
        response = await agent.invoke(user_query, session_id)
        print(f"Agent: {response}")


if __name__ == "__main__":
    import asyncio

    # Test a direct call to the Milvus search function for debugging
    # print("Directly testing Milvus search (ensure Milvus is running and collection exists):")
    # test_query = "What special offers are available for VIP members at the concert?"
    # direct_search_results = search_milvus_knowledge_base(test_query)
    # print(f"Direct search results for '{test_query}':\n{direct_search_results}\n")

    asyncio.run(main())
