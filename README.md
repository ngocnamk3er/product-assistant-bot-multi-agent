# 🤖 TellTimeAgent & Multi-Agent Demo – A2A with Google ADK

Welcome to **TellTimeAgent** and the **Multi-Agent** demo — a minimal Agent2Agent (A2A) implementation using Google's [Agent Development Kit (ADK)](https://github.com/google/agent-development-kit).

This example demonstrates how to build, serve, and interact with three A2A agents:
1. **TellTimeAgent** – replies with the current time.
2. **GreetingAgent** – fetches the time and generates a poetic greeting.
3. **OrchestratorAgent** – routes requests to the appropriate child agent.

All of them work together seamlessly via A2A discovery and JSON-RPC.

---

## 📦 Project Structure

```bash
version_3_multi_agent/
├── .env                         # Your GOOGLE_API_KEY (not committed)
├── pyproject.toml              # Dependency config
├── README.md                   # You are reading it!
├── app/
│   └── cmd/
│       └── cmd.py              # CLI to interact with the OrchestratorAgent
├── agents/
│   ├── greeting_agent/
│   │   ├── __main__.py         
│   │   ├── agent.py            
│   │   └── task_manager.py     
│   ├── sale_support_agent/
│   │   ├── __main__.py         
│   │   ├── agent.py            
│   │   └── task_manager.py     
│   ├── event_introduction_agent/
│   │   ├── __main__.py         
│   │   ├── agent.py            
│   │   └── task_manager.py     
│   ├── policy_qa_agent/
│   │   ├── __main__.py         
│   │   ├── agent.py            
│   │   └── task_manager.py     
│   └── host_agent/
│       ├── entry.py            
│       ├── orchestrator.py     
│       └── agent_connect.py    
├── server/
│   ├── server.py               # A2A JSON-RPC server implementation
│   └── task_manager.py         # Base in-memory task manager interface
└── utilities/
    ├── discovery.py            # Finds agents via `agent_registry.json`
    └── agent_registry.json     # List of child-agent URLs (one per line)
```

---

## 🛠️ Setup

1. **Clone & navigate**

    ```bash
    git clone https://github.com/ngocnamk3er/product-assistant-bot-multi-agent
    cd a2a_samples/version_3_multi_agent
    ```

2. **Create & activate a venv**

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3. **Install dependencies**

    Using [`uv`](https://github.com/astral-sh/uv):

    ```bash
    uv pip install .
    ```

    Or with pip directly:

    ```bash
    pip install .
    ```

4. **Set your API key**

    Create `.env` at the project root:
    ```bash
    echo "GOOGLE_API_KEY=your_api_key_here" > .env
    ```

---

## 🧱 Kiến trúc hệ thống

### A2A MCP Architecture
![A2A MCP Architecture](images/a2a_mcp_architecture.png)

### Chatbot App Architecture
![Chatbot App Architecture](images/chatbot_app_architecture.png)

## 📸 Demo minh họa

### Demo 1
![Demo 1](images/demo1.png)

### Demo 2
![Demo 2](images/demo2.png)

### Demo 3
![Demo 3](images/demo3.png)

### Demo 4
![Demo 4](images/demo4.png)

## 🎬 Demo Walkthrough

**Start the Server backend**
```bash
python -m backend.server --host localhost --port 2000
```

**Start the Phone QA agent**
```bash
python -m agents.sale_support_agent --host localhost --port 10005
```



**Start the Policy QA agent**
```bash
python -m agents.policy_qa_agent --host localhost --port 10004
```


**Start the Event introduction agent**
```bash
python -m agents.event_introduction_agent --host localhost --port 10003
```


**Start the GreetingAgent**
```bash
python -m agents.greeting_agent --host localhost --port 10001
```

**Start the Orchestrator (Host) Agent**
```bash
python -m agents.host_agent.entry --host localhost --port 10000
```

**Launch the CLI (cmd.py)**
```bash
python -m app.cmd.cmd --agent http://localhost:10002
```

**Try it out!**
```bash
> What time is it?
Agent says: The current time is: 2025-05-05 14:23:10

> Greet me
Agent says: Good afternoon, friend! The golden sun dips low...
```

---

## 🔍 How It Works

1. **Discovery**: OrchestratorAgent reads `utilities/agent_registry.json`, fetches each agent’s `/​.well-known/agent.json`.
2. **Routing**: Based on intent, the Orchestrator’s LLM calls its tools:
   - `list_agents()` → lists child-agent names
   - `delegate_task(agent_name, message)` → forwards tasks
3. **Child Agents**:
   - TellTimeAgent returns the current time.
   - GreetingAgent calls TellTimeAgent then crafts a poetic greeting.
4. **JSON-RPC**: All communication uses A2A JSON-RPC 2.0 over HTTP via Starlette & Uvicorn.

---

## 📖 Learn More

- A2A GitHub: https://github.com/google/A2A  
- Google ADK: https://github.com/google/agent-development-kit