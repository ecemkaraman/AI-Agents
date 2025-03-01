### **Building Code Agents in smolagents**

### **Why Code Agents?**

- Default agent type in **smolagents**—generates and executes **Python tool calls**.
- More **efficient, expressive, and reusable** than JSON-based actions.
- Works directly with complex **objects (e.g., images, data structures)**.
- Natural for **LLMs**—aligns with training data containing high-quality code.

---

### **How a Code Agent Works**

1️⃣ **Stores user query & system prompt** as memory.

2️⃣ **Writes tool calls in Python**, not JSON.

3️⃣ **Executes code directly**, logging results.

4️⃣ **Iterates in a loop** (Think → Act → Observe) until the task is complete.

---

### **Example: Alfred, the AI Butler**

💿 **Step 1: Create an Agent to Find Party Music**

```python
from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel

agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=HfApiModel())

agent.run("Search for the best music recommendations for a party at the Wayne's mansion.")

```

🎶 **Output:** Generates & executes Python code to fetch party playlists.

---

🍽 **Step 2: Create a Custom Tool for Menu Planning**

```python
from smolagents import CodeAgent, tool, HfApiModel

@tool
def suggest_menu(occasion: str) -> str:
    """Suggests a menu based on the occasion."""
    return {
        "casual": "Pizza, snacks, and drinks.",
        "formal": "3-course dinner with wine and dessert.",
        "superhero": "Buffet with high-energy and healthy food."
    }.get(occasion, "Custom menu for the butler.")

agent = CodeAgent(tools=[suggest_menu], model=HfApiModel())
agent.run("Prepare a formal menu for the party.")

```

🍷 **Output:** Suggests **custom menus** for different party themes.

---

⏳ **Step 3: Estimate Party Preparation Time Using Python Imports**

```python
from smolagents import CodeAgent, HfApiModel

agent = CodeAgent(tools=[], model=HfApiModel(), additional_authorized_imports=['datetime'])

agent.run("""
Alfred needs to prepare for the party. Tasks:
1. Drinks - 30 min
2. Decorate - 60 min
3. Menu setup - 45 min
4. Music - 45 min

If we start now, at what time will the party be ready?
""")

```

⌛ **Output:** Computes the exact party start time based on preparation tasks.

---

### **Sharing Agents on Hugging Face Hub**

🚀 **Publish your agent:**

```python
agent.push_to_hub('your_username/AlfredAgent')

```

🛠 **Reload and use anytime:**

```python
alfred_agent = agent.from_hub('your_username/AlfredAgent')
alfred_agent.run("Find the best playlist for a villain masquerade party.")

```

🎭 **Available in Hugging Face Spaces for real-time use!**

---

### **Enhancing Traceability with OpenTelemetry & Langfuse**

📡 **Track & debug agent runs with OpenTelemetry:**

```python
pip install opentelemetry-sdk opentelemetry-exporter-otlp openinference-instrumentation-smolagents

```

🔗 **Set up logging in Langfuse:**

```python
import os, base64

LANGFUSE_PUBLIC_KEY="your_public_key"
LANGFUSE_SECRET_KEY="your_secret_key"
LANGFUSE_AUTH=base64.b64encode(f"{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}".encode()).decode()

os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "<https://cloud.langfuse.com/api/public/otel>"
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {LANGFUSE_AUTH}"

```

📊 **Monitor agent actions in Langfuse dashboard.**

---

### **Why Code Agents Beat JSON-Based Actions**

| Feature | Code Agents | JSON Agents |
| --- | --- | --- |
| **Execution** | Runs Python code directly | Requires JSON parsing |
| **Flexibility** | Supports loops, conditionals, objects | Limited to predefined fields |
| **LLM Alignment** | Matches LLM training data | Requires extra parsing logic |
| **Composability** | Easily reuses code functions | Needs external orchestration |

💡 **Key Takeaway:** Code agents **streamline execution**, making AI workflows **faster & more reliable**.

---

🚀 **Next Steps:**

✅ **Build your own CodeAgent** with custom tools.

✅ **Push it to Hugging Face Hub** for easy sharing.

✅ **Monitor and optimize performance** using OpenTelemetry.

🎉 **Let’s build powerful AI Agents together!**
