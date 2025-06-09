# =============================================================================
# agents/host_agent/entry.py
# =============================================================================
# 🎯 Purpose:
# Boots up the OrchestratorAgent as an A2A server.
# Uses the shared registry file to discover all child agents,
# then delegates routing to the OrchestratorAgent via A2A JSON-RPC.
# =============================================================================
import asyncpg                              # Async PostgreSQL driver for database interactions
import asyncio                              # Built-in for running async coroutines
import logging                              # Standard Python logging module
import click                                # Library for building CLI interfaces
import os                                  # For reading environment variables
# Utility for discovering remote A2A agents from a local registry
from utilities.discovery import DiscoveryClient
# Shared A2A server implementation (Starlette + JSON-RPC)
from server.server import A2AServer
# Pydantic models for defining agent metadata (AgentCard, etc.)
from models.agent import AgentCard, AgentCapabilities, AgentSkill
from starlette.middleware import Middleware
from middleware.auth import AuthMiddleware
from database.database import Database  # Custom database connection manager
# Orchestrator implementation and its task manager
from agents.host_agent.orchestrator import (
    OrchestratorAgent,
    OrchestratorTaskManager
)
from functools import lru_cache

@lru_cache()
def get_settings():
    return {
        "database_url": os.getenv("DATABASE_URL"),
        "jwt_secret_key": os.getenv("JWT_SECRET_KEY"),
        "jwt_algorithm": os.getenv("JWT_ALGORITHM"),
        "access_token_expire_minutes": int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES")),
    }

SETTINGS = get_settings()

class Database:
    def __init__(self, db_url):
        self._db_url = db_url
        self.pool = None

    async def connect(self):
        if not self.pool:
            self.pool = await asyncpg.create_pool(dsn=self._db_url, min_size=1, max_size=10)
            print("Database connection pool created.")

    async def disconnect(self):
        if self.pool:
            await self.pool.close()
            print("Database connection pool closed.")

    async def fetch_one(self, query, *params):
        async with self.pool.acquire() as connection:
            return await connection.fetchrow(query, *params)
        
db = Database(SETTINGS["database_url"])

# Configure root logger to show INFO-level messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
@click.command()
@click.option(
    "--host", default="localhost",
    help="Host to bind the OrchestratorAgent server to"
)
@click.option(
    "--port", default=10000,
    help="Port for the OrchestratorAgent server"
)
@click.option(
    "--registry",
    default=None,
    help=(
        "Path to JSON file listing child-agent URLs. "
        "Defaults to utilities/agent_registry.json"
    )
)
def main(host: str, port: int, registry: str):
    """
    Entry point to start the OrchestratorAgent A2A server.

    Steps performed:
    1. Load child-agent URLs from the registry JSON file.
    2. Fetch each agent's metadata via `/.well-known/agent.json`.
    3. Instantiate an OrchestratorAgent with discovered AgentCards.
    4. Wrap it in an OrchestratorTaskManager for JSON-RPC handling.
    5. Launch the A2AServer to listen for incoming tasks.
    """
    # 1) Discover all registered child agents from the registry file
    discovery = DiscoveryClient(registry_file=registry)
    # Run the async discovery synchronously at startup
    agent_cards = asyncio.run(discovery.list_agent_cards())

    # Warn if no agents are found in the registry
    if not agent_cards:
        logger.warning(
            "No agents found in registry – the orchestrator will have nothing to call"
        )

    # 2) Define the OrchestratorAgent's own metadata for discovery
    capabilities = AgentCapabilities(streaming=False)
    skill = AgentSkill(
        id="orchestrate",                          # Unique skill identifier
        name="Orchestrate Tasks",                  # Human-friendly name
        description=(
            "Routes user requests to the appropriate child agent, "
            "based on intent (time, greeting, etc.)"
        ),
        tags=["routing", "orchestration"],       # Keywords to aid discovery
        examples=[                                  # Sample user queries
            "What is the time?",
            "Greet me",
            "Say hello based on time"
        ]
    )
    orchestrator_card = AgentCard(
        name="OrchestratorAgent",
        description="Delegates tasks to discovered child agents",
        url=f"http://{host}:{port}/",             # Public endpoint
        version="1.0.0",
        defaultInputModes=["text"],                # Supported input modes
        defaultOutputModes=["text"],               # Supported output modes
        capabilities=capabilities,
        skills=[skill]
    )

    # 3) Instantiate the OrchestratorAgent and its TaskManager
    orchestrator = OrchestratorAgent(agent_cards=agent_cards)
    task_manager = OrchestratorTaskManager(agent=orchestrator)


    # 4) Create and start the A2A server
    
    middleware = [
        Middleware(AuthMiddleware, db=db, settings=SETTINGS)
    ]
    server = A2AServer(
        host=host,
        port=port,
        agent_card=orchestrator_card,
        task_manager=task_manager
    )
    server.app.add_middleware(*middleware)
    server.app.on_startup.append(db.connect)
    server.app.on_shutdown.append(db.disconnect)
    server.start()


if __name__ == "__main__":
    main()
