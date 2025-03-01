### **Writing Actions as Code or JSON in smolagents**

### **Code Agents vs Tool Calling Agents**

🔹 **Code Agents** → Generate and execute **Python snippets**.

🔹 **ToolCallingAgents** → Use **LLM tool-calling APIs** to produce **JSON-based tool calls**.

🔹 **Code = More flexibility**, JSON = **Simpler workflows**.

---

### **Example: Searching for Catering Services**

👨‍🍳 **CodeAgent Example (Python Execution)**

```python
for query in [
    "Best catering services in Gotham City",
    "Party theme ideas for superheroes"
]:
    print(web_search(f"Search for: {query}"))

```

🔹 **Executes Python directly**

🔹 **Handles variables & logic dynamically**

---

📜 **ToolCallingAgent Example (JSON-Based Calls)**

```json
[
    {"name": "web_search", "arguments": "Best catering services in Gotham City"},
    {"name": "web_search", "arguments": "Party theme ideas for superheroes"}
]

```

🔹 **LLM generates JSON instructions**

🔹 **Requires parsing before execution**

---

### **How Do Tool Calling Agents Work?**

1️⃣ **Multi-Step Process:** Similar to Code Agents but with JSON-based tool calls.

2️⃣ **LLM Outputs JSON:** Instead of Python, generates structured tool requests.

3️⃣ **System Parses & Executes:** Extracts tool names & arguments, runs the appropriate function.

---

### **Example: Running a Tool Calling Agent**

🎵 **Find Party Music with DuckDuckGo**

```python
from smolagents import ToolCallingAgent, DuckDuckGoSearchTool, HfApiModel

agent = ToolCallingAgent(tools=[DuckDuckGoSearchTool()], model=HfApiModel())

agent.run("Search for the best music recommendations for a party at the Wayne's mansion.")

```

🖥 **ToolCallingAgent Output:**

```
╭──────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Calling tool: 'web_search' with arguments: {'query': "best music recommendations for a party at Wayne's  │
│ mansion"}                                                                                                │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────╯

```

🔹 **Doesn’t execute Python directly**

🔹 **Passes tool calls as JSON for execution**

---

### **Code vs JSON: Which One to Use?**

| Feature | **Code Agents** 🖥 | **ToolCallingAgents** 📜 |
| --- | --- | --- |
| **Execution** | Runs Python code | Parses & executes JSON |
| **Use Case** | Complex workflows, dynamic logic | Simple function calls |
| **Flexibility** | High (loops, logic, conditionals) | Limited to predefined JSON structure |
| **LLM Compatibility** | Uses Python from training data | Uses LLM's built-in tool-calling APIs |
| **Integration** | Works with any LLM or API | Requires LLM providers with function calling |

💡 **Use CodeAgents for advanced workflows, ToolCallingAgents for lightweight, structured tasks!**

---

🚀 **Next Steps:**

✅ Choose the right agent for your use case.

✅ Experiment with both agent types in **smolagents**.

✅ Continue refining **Alfred’s AI-powered party planner! 🎉**
