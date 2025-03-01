### **When to Use an Agentic Framework**

- Not always needed—use **only** when workflows become complex (e.g., function calling, multi-agent systems).
- Simple **prompt chaining** can be done with plain code for better control and fewer abstractions.
- When workflows scale, key features required:
    - **LLM Engine**
    - **Tool Access**
    - **Parser for Tool Calls**
    - **System Prompt Sync**
    - **Memory & Logging**

---

### **smolagents: Lightweight AI Agent Framework**

### **Why smolagents?**

- **Open-source**, **minimal** framework for AI agents.
- Focuses on **CodeAgents**—agents that generate and execute Python code.
- Supports **ToolCallingAgents**—agents using JSON/text-based tool calls.
- Simple integration with **Hugging Face**, **Gradio**, and external APIs.

---

### **Key Features Covered in This Unit**

### **1️⃣ CodeAgents**

- Generate & execute Python code instead of structured JSON outputs.
- Simplifies execution—directly runs the code instead of parsing JSON.

### **2️⃣ ToolCallingAgents**

- Calls tools via **JSON/text blobs** instead of generating executable code.
- Requires parsing to extract tool calls from LLM outputs.

### **3️⃣ Tools**

- Functions that enable LLMs to perform actions.
- Defined via **@tool decorator** or **Tool class**.
- Explore built-in & community-contributed tools.

### **4️⃣ Retrieval Agents**

- Enable **knowledge search & retrieval** from external sources.
- Use **vector stores** and **RAG (Retrieval-Augmented Generation)** for smarter AI agents.

### **5️⃣ Multi-Agent Systems**

- Combine multiple agents to handle **complex** workflows.
- Example: **Web search + Code execution agent** working together.

### **6️⃣ Vision & Browser Agents**

- Use **Vision-Language Models (VLMs)** to analyze images.
- Build agents that can **browse the web & extract information**.

---

### **Why Choose smolagents?**

- **Lightweight**: Minimal setup, fast experimentation.
- **Code-First**: Direct execution of tool calls (no JSON parsing).
- **Flexible Model Support**: Works with **Hugging Face APIs**, **LiteLLM**, **Azure**, **OpenAI API**, etc.
- **HF Hub Integration**: Uses Gradio Spaces & external tools seamlessly.

---

### **smolagents vs Other Agentic Frameworks**

| Feature | smolagents | LlamaIndex | LangGraph |
| --- | --- | --- | --- |
| **Agent Type** | Code & ToolCalling Agents | Retrieval-focused Agents | Graph-based Agents |
| **Execution Format** | **Python Code** | JSON & Text Outputs | JSON & Graph Nodes |
| **Tooling Support** | **@tool decorator**, **Tool class** | External APIs, DBs | Structured function calls |
| **Best Use Case** | **Lightweight AI agents, direct execution** | **Document retrieval & search** | **Complex decision workflows** |

---

### **Model Integration in smolagents**

Supports multiple model backends for **LLM execution**:

- **TransformersModel** – Local Hugging Face pipelines.
- **HfApiModel** – Hugging Face **serverless inference**.
- **LiteLLMModel** – Lightweight API calls.
- **OpenAIServerModel** – OpenAI API integration.
- **AzureOpenAIServerModel** – Azure-based deployments.

---

### **When to Use smolagents?**

✅ **If you need:**

✔️ **Fast prototyping** with minimal complexity.

✔️ **Code-driven** AI workflows (Python-first execution).

✔️ **Flexible** agent design (CodeAgent + ToolCallingAgent).

🚫 **Not ideal for:**

❌ **Highly structured workflows** (LangGraph is better).

❌ **Complex multi-step decision-making** (LlamaIndex might fit better).
