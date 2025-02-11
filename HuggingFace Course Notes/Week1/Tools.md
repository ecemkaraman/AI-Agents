# Tools in AI Agents

### **1. What Are AI Tools?**

- **Definition**: A Tool is a **function** given to the **LLM** to fulfill a **clear objective**
- **Purpose**: Enables AI agents to take actions beyond text generation.
- A good tool should **complement the power of an LLM**.
- **Common Tools**:
    - **Web Search** – Fetches real-time information.
    - **Image Generation** – Creates images from text.
    - **Retrieval** – Accesses external data sources.
    - **API Interface** – Interacts with services (GitHub, YouTube, etc.).
- **Why They Matter**:
    - LLMs have knowledge limitations (cutoff dates, training data constraints).
    - Tools extend functionality to provide real-time, external, or specialized data.
    - Example: Instead of guessing today’s weather, an LLM can use a weather API tool.

---

### **2. Components of a Tool**

A well-defined tool consists of:

- **Name** + **Description**– Clearly indicates function + what it does
- **Inputs** – Defines required arguments and types.
- **Outputs** – Specifies expected return values. (optional)
- **Callable** – A function that executes an action.

```python
def calculator(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b
```

- **Name**: `calculator`
- **Description**: Multiplies two integers.
- **Inputs**: `a (int)`, `b (int)`
- **Output**: `int` (product of `a` and `b`)

---

### **3. How LLMs Use Tools**

- LLMs can’t call tools directly; they generate text that instructs an Agent to invoke the necessary tool based on the prompt.
- The Agent parses this output, executes the tool, and returns the result.
- Tool calls function as hidden conversation messages.
- The Agent processes the request, retrieves outputs, updates the conversation, and sends it back to the LLM for the final response.
- **Example (Weather Query Workflow)**:
    - User asks: "What’s the weather in Paris?"
    - LLM generates: `call weather_api("Paris")`
    - Agent runs the API call.
    - API returns: `"12°C, cloudy"`
    - Agent feeds response back into LLM, which formats it for the user.

---

### **4. Providing Tools to an LLM**

- **System Message**: Tools are introduced through precise textual descriptions of available tools.
- **Essential Details**:
    - **Functionality** – What the tool does.
    - **Expected Inputs** – Data format and type.
    - **Expected Outputs** – Return value type.
- **Format Example** (for an LLM system prompt):
    
    ```python
    Tool Name: calculator, 
    Description: Multiply two integers., 
    Arguments: a: int, b: int, 
    Outputs: int
    ```
    
- **Why Structure Matters**:
    - LLMs need a consistent format to recognize and use tools.
    - Common formats: JSON, structured text, or inline code.

---

### **5. Automating Tool Definition with Python**

- **Problem**: Manually writing descriptions can be error-prone.
- **Solution**: Use Python introspection to extract function metadata.→use type hints, docstrings, and sensible function names

**3 Fundamental Ways to Define & Manage Tools in Python for AI Agents:**

- **1️⃣ Manual Definition (Dicts or Strings)** - small scale
    - **🔹 Simple but static**: Tools are manually defined using dictionaries or structured strings.
    - **🔹 No automation**: Every tool must be explicitly written out.
    - **🔹 Best for**: Small-scale setups or one-off tool definitions.
    
    ```python
    tool_description = {
        "name": "calculator",
        "description": "Multiply two integers.",
        "arguments": {"a": "int", "b": "int"},
        "output": "int"
    }
    
    def calculator(a, b):
        return a * b
    ```
    
    ✅ **Pros**: Explicit control, easy to debug
    
    ❌ **Cons**: Repetitive, not scalable
    

---

- **2️⃣ Decorators (`@tool`)** - automate w/ min complexity
    - **🔹 Automates metadata extraction**: Uses decorators to generate tool descriptions.
    - **🔹 Tightly coupled**: Metadata is attached directly to the function.
    - **🔹 Best for**: Small to medium-scale projects where automation is needed.
    
    ```python
    def tool(func):
        """Decorator to extract function metadata for AI tool usage."""
        func.tool_name = func.__name__
        func.description = func.__doc__
        func.arguments = func.__annotations__
        func.output = func.arguments.pop("return", "Unknown")
        return func
    
    @tool
    def calculator(a: int, b: int) -> int:
        """Multiply two integers."""
        return a * b
    
    print(calculator.tool_name)  # "calculator"
    print(calculator.arguments)  # {'a': 'int', 'b': 'int'}
    ```
    
    ✅ **Pros**: Clean syntax, metadata auto-extraction
    
    ❌ **Cons**: Less flexible, function-bound
    

---

- **3️⃣ `Tool` Class (OOP)** - large scale, reusable for any tool
    
    **🔹 Encapsulates tool logic & metadata**: Enables structured and reusable tool management.
    
    **🔹  Reusability & Consistency**: Can be used for multiple tools & use standard format  
    
    **🔹 Allows dynamic registration**: Tools can be stored, added, and modified at runtime.
    
    **🔹 Execution Control**: The `__call__()` method allows tools to be invoked dynamically.
    
    **🔹 Best for**: Large-scale applications with multiple tools.
    
    ✅ **Pros**: Scalable, reusable, supports dynamic registration
    
    ❌ **Cons**: Slightly more complex
    
    ```python
    class Tool:
        """
        A class representing a reusable piece of code (Tool).
        
        Attributes:
            name (str): Name of the tool.
            description (str): A textual description of what the tool does.
            func (callable): The function this tool wraps.
            arguments (list): A list of argument.
            outputs (str or list): The return type(s) of the wrapped function.
        """
    
        def __init__(self, name: str, description: str, func: callable, arguments: dict, output: str):
            self.name = name
            self.description = description
            self.func = func
            self.arguments = arguments
            self.output = output
    
        def to_string(self) -> str:
            """
            Generate a structured representation of the tool.
            """
            args_str = ", ".join([f"{arg}: {type_}" for arg, type_ in self.arguments.items()])
            return f"Tool Name: {self.name}, Description: {self.description}, Arguments: {args_str}, Output: {self.output}"
    
        def __call__(self, *args, **kwargs):
            """
            Invoke the tool function with given arguments.
            """
            return self.func(*args, **kwargs)
    
    @Tool
    def calculator(a: int, b: int) -> int:
        """Multiply two integers."""
        return a * b
    
    print(calculator.to_string())
    print(calculator(3, 4))  # Calls tool dynamically
    ```
    
    - 2 Usage Options for `Tool` class: decorator-based (BP) or manual registration
    
    ```python
    # **Decorator-Based Tool Definition**
    @Tool(
        name="Calculator",
        description="Multiplies two integers.",
        arguments={"a": "int", "b": "int"},
        output="int"
    )
    def multiply(a: int, b: int) -> int:
        """Multiply two integers."""
        return a * b
    
    print(multiply.to_string())  # Metadata
    print(multiply.execute(4, 5))  # Output: 20
    ```
    
    ```python
    # **Manual Registration**
    def multiply(a: int, b: int) -> int:
        """Multiply two integers."""
        return a * b
    
    multiply_tool = Tool(
        name="Calculator",
        description="Multiplies two integers.",
        arguments={"a": "int", "b": "int"},
        output="int",
        func=multiply
    )
    
    print(multiply_tool.to_string())  # Metadata
    print(multiply_tool.execute(4, 5))  # Output: 20
    ```
    

---

**When to Use Each Approach**

| **Scenario** | **Best Approach** |
| --- | --- |
| **Few tools, simple setup** | **Manual dictionary-based definition** |
| **Want automatic metadata extraction for functions** | **Decorators (`@tool`)** |
| **Need flexibility, dynamic registration, or tool execution management** | **Tool Class (`Tool(func)`)** |
- **Arguments: Implicit(Dunder/Magic) Attributes vs Explicit**
    
    <aside>
    
    **BP:** Use explicit arguments for AI Agents
    
    </aside>
    
    - **Explicit arguments** → More control, ensures consistency, works with any callable, avoids dependency on function annotations, flexible for APIs/tools, but requires manual input.
    - **Auto-extracted annotations** → Auto-extracts function metadata, less manual work, requires type hints, less flexible→may not always align with intended tool behavior.
        - `func.__name__` → **Tool Name**
        - `func.__doc__` → **Function (tool) docstring**
        - `func.__annotations__` → **Explicit Argument Types & Return Type**
        
        ```python
        def __init__(self, func):
                self.name = func.__name__
                self.description = func.__doc__ or "No description"
                self.arguments = func.__annotations__
                self.output = self.arguments.pop("return", "Unknown")
                self.func = func
        ```
        

---

### **6. Why Tools Are Essential**

- **Extend LLM Capabilities**: Perform actions beyond text prediction.
- **Enable Real-Time Data**: Access updated information via APIs.
- **Improve Accuracy**: Reduce hallucinations by integrating factual sources.
- **Enhance Usability**: Provide structured responses for specialized tasks.

### **Summary**

✔ **Tools = Functions that extend LLM capabilities**

✔ **LLMs generate tool calls as text; Agents execute them**

✔ **System messages define available tools for the model**

✔ **Python introspection automates tool description generation**

✔ **Structured formats ensure tool reliability and usability**

🚀 **Next Step**: Implementing AI Agents with tool integrations.
