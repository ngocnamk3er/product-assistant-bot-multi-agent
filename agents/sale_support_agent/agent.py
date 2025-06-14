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
from google import genai

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


# MCP server
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService # Optional
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, SseServerParams, StdioServerParameters

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
TEXT_CONTENT_FIELD_NAMES = ["name", "formatted_specs", "price"]
SOURCE_FIELD_NAME = "url"  # <<< TÙY CHỌN: TÊN TRƯỜNG CHỨA NGUỒN GỐC
TOP_K_RESULTS = 5
GENAI_MODEL_NAME = "gemini-2.5-flash-preview-05-20"  # Model for LLM tasks

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




# -----------------------------------------------------------------------------
# 🕒 EventIntroductionAgent: Your AI agent that tells the time and searches KB
# -----------------------------------------------------------------------------


class SaleSupportAgent:
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

        toolset = MCPToolset(
            # Use StdioServerParameters for local process communication
            connection_params=SseServerParams(url="http://localhost:30000/sse"),
        )

        # Create FunctionTool for Milvus search
        # The ADK will infer the schema from the function's signature and docstring
        # milvus_kb_tool = FunctionTool(search_milvus_knowledge_base)
        return LlmAgent(
            model="gemini-2.5-flash-preview-05-20",
            name="phone_question_answer_agent",
            description="Trợ lý AI chuyên nghiệp hỗ trợ việc tư vấn sản phầm và đặt hàng điện thoại di động.",
            # instruction=(
            #     "Bạn là trợ lý AI chuyên về trả lời các câu hỏi liên quan về điện thoại di động, thực hiện theo các trường hợp sau "
            #     "Trường hợp 1: Nếu câu hỏi của người dùng chỉ liên quan duy nhất đến việc tìm kiếm các sản phẩm điện thoại di động, "
            #     "Bước 1: Tách từ truy vấn hợp lí từ câu hỏi của người dùng, ví dụ: "
            #     '"Tìm cho tôi thông tin về Iphone 15" sẽ được tách thành "Iphone 15". '
            #     'Bước 2: Tìm kiếm trong cơ sở dữ liệu Milvus bằng tool "search_milvus_knowledge_base" để lấy thông tin về sản phẩm điện thoại di động phù hợp với truy vấn. '
            #     "Bước 3: Trả lời người dùng bằng cách cung cấp thông tin chi tiết về sản phẩm, bao gồm tên, thông số kỹ thuật và giá cả. "
            #     "Trường hợp 2: Nếu câu hỏi của người dùng liên quan đến việc Create, Read, Update, Delete (CRUD) dữ liệu trong cơ sở dữ liệu, "
            #     "Bước 1: Tách từ truy vấn hợp lí từ câu hỏi của người dùng, ví dụ: "
            #     'Bước 2:Nếu cảm thấy cần thiết thì Tìm kiếm trong cơ sở dữ liệu Milvus bằng tool "search_milvus_knowledge_base" để lấy thông tin về sản phẩm điện thoại di động phù hợp với truy vấn. '
            #     "Bước 3: Sử dụng tool 'generate_sql_from_schema_and_question' để chuyển đổi câu hỏi của người dùng thành câu lệnh SQL phù hợp với cơ sở dữ liệu. "
            #     "Bước 4: Trả lời người dùng bằng cách cung cấp câu lệnh SQL đã chuyển đổi. "
            # ),
            instruction=(
                "Bạn là trợ lý AI chuyên về trả lời các câu hỏi liên quan về điện thoại di động, thực hiện theo các trường hợp sau "
                "Nếu câu hỏi của người dùng liên quan đến việc tìm kiếm các sản phẩm điện thoại di động, "
                "Bước 1: Tách từ truy vấn hợp lí từ câu hỏi của người dùng, ví dụ: "
                '"Tìm cho tôi thông tin về Iphone 15" sẽ được tách thành "Iphone 15". '
                'Bước 2: Tìm kiếm trong cơ sở dữ liệu Milvus bằng tool "search_milvus_knowledge_base" để lấy thông tin về sản phẩm điện thoại di động phù hợp với truy vấn. '
                "Bước 3: Trả lời người dùng"
            ),
            tools=[
                toolset,
                # milvus_kb_tool,
                # converted_user_question_to_sql_tool,
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
    agent = SaleSupportAgent()
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
