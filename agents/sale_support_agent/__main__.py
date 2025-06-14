# =============================================================================
# agents/google_adk/__main__.py
# =============================================================================
# Purpose:
# This is the main script that starts your TellTimeAgent server.
# It:
# - Declares the agent’s capabilities and skills
# - Sets up the A2A server with a task manager and agent
# - Starts listening on a specified host and port
#
# This script can be run directly from the command line:
#     python -m agents.google_adk
# =============================================================================

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os              # For reading environment variables
# Your custom A2A server class
from server.server import A2AServer

# Models for describing agent capabilities and metadata
from models.agent import AgentCard, AgentCapabilities, AgentSkill

# Task manager and agent logic
from agents.sale_support_agent.task_manager import SaleSupportTaskManager
from agents.sale_support_agent.agent import SaleSupportAgent

# CLI and logging support
import click           # For creating a clean command-line interface
import logging         # For logging errors and info to the console


# -----------------------------------------------------------------------------
# Setup logging to print info to the console
# -----------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Main Entry Function – Configurable via CLI
# -----------------------------------------------------------------------------

@click.command()
@click.option("--host", default="localhost", help="Host to bind the server to")
@click.option("--port", default=10004, help="Port number for the server")
def main(host, port):
    """
    This function sets up everything needed to start the agent server.
    You can run it via: `python -m agents.google_adk --host 0.0.0.0 --port 12345`
    """
    public_hostname = os.getenv("AGENT_PUBLIC_HOSTNAME", host)
    # Define what this agent can do – in this case, it does NOT support streaming
    capabilities = AgentCapabilities(streaming=False)

    # Define the skill this agent offers (used in directories and UIs)
    skill = AgentSkill(
        id="phone_question_answering_agent",                                 # Unique skill ID
        name="Phone Question Answering Agent",                          # Human-friendly name
        description="Trả lời người dùng về các sản phẩm điện thoại hiện có",    # What the skill does
        examples=["Cho tôi thông tin về điện thoại Iphone 16 mới ra mắt?", "Iphone 16 mới có các tính năng...."]  # Example queries
    )

    # Create an agent card describing this agent’s identity and metadata
    agent_card = AgentCard(
        name="Phone Question Answering Agent",                               # Name of the agent
        description="Trả lời người dùng về các sản phẩm điện thoại",  # Description
        url=f"http://{public_hostname}:{port}/",                       # The public URL where this agent lives
        version="1.0.0",                                    # Version number
        defaultInputModes=SaleSupportAgent.SUPPORTED_CONTENT_TYPES,  # Input types this agent supports
        defaultOutputModes=SaleSupportAgent.SUPPORTED_CONTENT_TYPES, # Output types it produces
        capabilities=capabilities,                          # Supported features (e.g., streaming)
        skills=[skill]                                      # List of skills it supports
    )

    server = A2AServer(
        host=host,
        port=port,
        agent_card=agent_card,
        task_manager=SaleSupportTaskManager(agent=SaleSupportAgent())
    )

    # Start listening for tasks
    server.start()


# -----------------------------------------------------------------------------
# This runs only when executing the script directly via `python -m`
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
