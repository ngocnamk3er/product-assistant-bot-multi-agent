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
    "phone_knowledge_base"  # <<< THAY THẾ BẰNG TÊN COLLECTION CỦA BẠN
)
EMBEDDING_MODEL_NAME = "embedding-001"  # Or "models/text-embedding-004" etc.
VECTOR_FIELD_NAME = (
    "embedding"  # <<< THAY THẾ BẰNG TÊN TRƯỜNG VECTOR TRONG COLLECTION CỦA BẠN
)
TEXT_CONTENT_FIELD_NAMES = [
    "name",
    "formatted_specs",
    "price"
]
SOURCE_FIELD_NAME = "url"  # <<< TÙY CHỌN: TÊN TRƯỜNG CHỨA NGUỒN GỐC
TOP_K_RESULTS = 5

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
                output_dimensionality=768,  # Optional
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
            "metric_type": "COSINE",  # Or "IP" (Inner Product) depending on your data/preference
            "params": {
                "nprobe": 10
            },  # Example search param, adjust based on your index type
        }

        output_fields = TEXT_CONTENT_FIELD_NAMES
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
             formatted_results += f"\nResult {i+1} (Score: {hit.distance:.4f}):\n"

        # SỬA Ở ĐÂY: Lấy từng trường trong TEXT_CONTENT_FIELD_NAMES
        content_parts = []
        for field_name in TEXT_CONTENT_FIELD_NAMES:
            value = hit.entity.get(field_name, "N/A")
            # Tùy chỉnh cách hiển thị tên trường nếu muốn (ví dụ: viết hoa chữ cái đầu)
            display_field_name = field_name.replace("_", " ").capitalize()
            content_parts.append(f"{display_field_name}: {value}")
            content_str = "\n".join(content_parts)

            formatted_results += f"Content:\n{content_str}\n"
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


class PhoneQuestionAnsweringAgent:
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
            model="gemini-2.5-flash-preview-05-20",
            name="phone_policy_question_answer_agent",
            description="Trợ lý AI chuyên cung cấp thông tin  điện thoại trong cơ sỡ diệu liệu của công ty",
            instruction=(
                "Bạn là trợ lý AI chuyên về trả lời các câu hỏi liên quan về điện thoại di động, thực hiện theo các bước sau "
                "Bước 1: không lấy hết câu hỏi người dùng mà phải tách từ khóa từ câu hỏi của người dùng để làm từ truy vấn"
                "Bước 2: Sử dụng công cụ 'search_milvus_knowledge_base' để tìm kiếm thông tin trong cơ sở dữ liệu."
                "Bước 3: Dựa trên kết quả tìm kiếm, trả lời câu hỏi của người dùng một cách chính xác và đầy đủ."
                "Nếu công cụ tìm kiếm không trả về thông tin liên quan, hãy thông báo cho người dùng rằng bạn không tìm thấy thông tin trong cơ sở dữ liệu."
            ),
            tools=[
                milvus_kb_tool,  # Add the new Milvus tool
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
    agent = PhoneQuestionAnsweringAgent()
    session_id = "test_session_milvus_001"

    print("Agent initialized. Type 'quit' to exit.")


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
    asyncio.run(main())
