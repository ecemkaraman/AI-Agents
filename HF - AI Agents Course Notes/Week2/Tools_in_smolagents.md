### **Tools in smolagents: Expanding Agent Capabilities**

💡 **Tools** = Functions that an **LLM-powered agent** can call to perform actions.

🔹 **Key components of a tool:**

- **Name** → Identifier for the tool.
- **Description** → Explains its function.
- **Inputs & Outputs** → Define arguments & return types.

---

### **🔧 Defining Tools in smolagents**

### **1️⃣ Using the `@tool` Decorator (Recommended for Simple Tools)**

✅ **Requires:**

- **Clear function name & docstring**
- **Type hints for inputs & outputs**
- **Explicit argument descriptions**

📌 **Example: Finding the Best Catering Service**

```python
from smolagents import CodeAgent, HfApiModel, tool

@tool
def catering_service_tool(query: str) -> str:
    """Returns the highest-rated catering service in Gotham City.

    Args:
        query: Search term for catering services.
    """
    services = {"Gotham Catering Co.": 4.9, "Wayne Manor Catering": 4.8, "Gotham City Events": 4.7}
    return max(services, key=services.get)

agent = CodeAgent(tools=[catering_service_tool], model=HfApiModel())

result = agent.run("Find the highest-rated catering service in Gotham City.")
print(result)  # Output: "Gotham Catering Co."

```

---

### **2️⃣ Using a Class-Based `Tool` (For Complex Tools)**

📌 **Example: Generating Superhero-Themed Party Ideas**

```python
from smolagents import Tool, CodeAgent, HfApiModel

class SuperheroPartyThemeTool(Tool):
    name = "superhero_party_theme_generator"
    description = "Suggests superhero party themes."

    inputs = {"category": {"type": "string", "description": "Type of superhero party."}}
    output_type = "string"

    def forward(self, category: str):
        themes = {
            "classic heroes": "Justice League Gala: Guests dressed as DC heroes.",
            "villain masquerade": "Gotham Rogues' Ball: A villain-themed masquerade.",
            "futuristic Gotham": "Neo-Gotham Night: Cyberpunk Batman Beyond theme."
        }
        return themes.get(category.lower(), "Try 'classic heroes', 'villain masquerade', or 'futuristic Gotham'.")

# Use the tool in an agent
party_theme_tool = SuperheroPartyThemeTool()
agent = CodeAgent(tools=[party_theme_tool], model=HfApiModel())

result = agent.run("Suggest a superhero party theme for a 'villain masquerade'.")
print(result)  # Output: "Gotham Rogues' Ball: A mysterious masquerade for Batman villains."

```

---

### **🔩 Default Toolbox in smolagents**

smolagents includes **pre-built tools** ready for use:

- 🔍 **DuckDuckGoSearchTool** → Web search.
- 🏗 **FinalAnswerTool** → Generates final responses.
- 🖥 **PythonInterpreterTool** → Executes Python code.
- 🔎 **GoogleSearchTool** → Google-based web search.
- 🌐 **VisitWebpageTool** → Fetches webpage data.

📌 **Example: Using Built-in Tools for Alfred's Party**

```python
agent = CodeAgent(
    tools=[DuckDuckGoSearchTool(), GoogleSearchTool(), FinalAnswerTool()],
    model=HfApiModel()
)

```

---

### **📤 Sharing & Importing Tools**

### **1️⃣ Share a Tool to the Hugging Face Hub**

```python
party_theme_tool.push_to_hub("{your_username}/party_theme_tool", token="<YOUR_HF_API_TOKEN>")

```

### **2️⃣ Import a Tool from the Hub**

```python
from smolagents import load_tool, CodeAgent, HfApiModel

image_gen_tool = load_tool("m-ric/text-to-image", trust_remote_code=True)

agent = CodeAgent(tools=[image_gen_tool], model=HfApiModel())
agent.run("Generate an image of a superhero party at Wayne Manor.")

```

### **3️⃣ Import a Hugging Face Space as a Tool**

```python
from smolagents import CodeAgent, HfApiModel, Tool

image_tool = Tool.from_space("black-forest-labs/FLUX.1-schnell", name="image_generator")

agent = CodeAgent(tools=[image_tool], model=HfApiModel())
agent.run("Generate an image for a Gotham superhero gala.")

```

### **4️⃣ Import a LangChain Tool**

```python
from langchain.agents import load_tools
from smolagents import CodeAgent, HfApiModel, Tool

search_tool = Tool.from_langchain(load_tools(["serpapi"])[0])

agent = CodeAgent(tools=[search_tool], model=HfApiModel())
agent.run("Find luxury entertainment for a superhero-themed event.")

```

---

### **🚀 Key Takeaways**

✅ **Use `@tool` decorator** for simple function-based tools.

✅ **Use `Tool` class** for complex tools with metadata.

✅ **Leverage built-in smolagents tools** for common tasks.

✅ **Easily share & import tools** from Hugging Face & LangChain.

🔗 **Next: Integrating tools into AI agent workflows for enhanced decision-making.**
