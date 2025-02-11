## AI Agent Workflow: Thought-Action-Observation Cycle

### **Core Concept**

- AI Agents operate in a continuous **Thought → Action → Observation** loop until they achieve their objective.
- This cycle ensures reasoning, execution via tools, and adaptation based on feedback.

---

### **1️⃣ Thought-Action-Observation Breakdown**

- **Thought:** The LLM decides the next step based on the input.
- **Action:** The agent calls a tool with the necessary parameters.
- **Observation:** The agent reflects on the tool's output to refine its response.

**Loop Execution:**

- The cycle repeats in a **while-loop** until the agent fulfills the user request.
- System prompts define agent behavior, tool access, and adherence to the Thought-Action-Observation cycle.

![image.png](attachment:b4d279b6-f7a0-4f84-9fb1-bd35f2d34ee4:image.png)

---

### **2️⃣ Example: Alfred, the Weather Agent**

**User Query:** *“What’s the weather like in New York today?”*

**Cycle Breakdown:**

1. **Thought (Internal Reasoning)**
    - Identifies the need for real-time weather data.
    - Determines that calling the `get_weather` tool is the next step.
2. **Action (Tool Invocation)**
    - Calls the weather API using a JSON-formatted request:
        
        ```json
        {
          "action": "get_weather",
          "action_input": { "location": "New York" }
        }
        ```
        
3. **Observation (Processing API Output)**
    - Receives response: *"Current weather in New York: partly cloudy, 15°C, 60% humidity."*
    - Adds observation to its prompt context.
4. **Updated Thought (Reflection)**
    - Confirms data retrieval + Prepares the final response.
5. **Final Action (User Response)**
    - Constructs and returns:*“The current weather in New York is partly cloudy with a temperature of 15°C and 60% humidity.”*

---

### **3️⃣ Key Takeaways**

✅ **Iterative Execution:** The agent loops through Thought-Action-Observation until the task is complete.

✅ **Tool Integration:** Real-time data retrieval enhances dynamic capabilities.

✅ **Adaptive Reasoning:** The agent refines responses based on fresh observations.

This **ReAct** cycle (Reasoning + Acting) enables AI Agents to iteratively solve complex tasks with continuous feedback. 🚀

---

## **Thought: Internal Reasoning & The Re-Act Approach**

### **1️⃣ Thought Process in AI Agents**

- **Purpose:** Internal reasoning enables agents to analyze information, break down problems, and strategize actions.
- **Role:** Uses the LLM's capacity to process prompts and decide next steps.
- **Function:** Allows agents to refine their approach based on observations, memory, and goals.
- **Types of Thought:**
    - **Planning:** “Break task into steps: gather data → analyze trends → generate report.”
    - **Analysis:** “Error suggests database connection issue.”
    - **Decision Making:** “User’s budget limits choice to mid-tier option.”
    - **Problem Solving:** “Profiling needed to find bottlenecks in code.”
    - **Memory Integration:** “User prefers Python—provide examples in Python.”
    - **Self-Reflection:** “Last approach failed—try a different strategy.”
    - **Goal Setting:** “Define acceptance criteria before starting.”
    - **Prioritization:** “Fix security vulnerability before adding features.”
- **Optional in Function-Calling LLMs:** Some LLMs, when fine-tuned for function-calling, bypass explicit thought processes.

---

### **2️⃣ The Re-Act Approach: Step-by-Step Reasoning**

- **Concept:** *ReAct = Reasoning (Think) + Acting (Act)*
- **Technique:** Uses prompting strategy like *“Let’s think step by step”* before generating an output.
- **Benefit:**
    - Encourages LLMs to **decompose problems** into smaller sub-tasks.
    - **Reduces errors** by focusing on logical sequencing rather than jumping to conclusions.
- **Example:**
    - Without ReAct: *“What is 34 × 47?”* → *“1598”*
    - With ReAct: *“Let’s think step by step.”*
        - *34 × 40 = 1360*
        - *34 × 7 = 238*
        - *Final result: 1598*
- **ReAct in Advanced Models:**
    - DeepSeek R1 & OpenAI’s o1 fine-tuned to "think before answering."
    - Use **structured thought sections** (`<think>` … `</think>`) in training, not just prompting (ReAct).

---

### ReAct vs Chain of Thought

- **ReAct includes CoT** as a **step within its process**, but extends it by enabling **interaction with external tools.** 🚀
- **CoT = Structured reasoning → One-time final answer.**
- **ReAct = Iterative reasoning + action → Real-world AI agents.**

| **Feature** | **Chain of Thought (CoT)** | **ReAct (Reasoning + Acting)** |
| --- | --- | --- |
| **Main Idea** | Step-by-step logical breakdown | Combines reasoning with tool usage |
| **Use Case** | Math, logic, problem-solving | AI agents, real-world interactions |
| **Execution** | Pure thought before final answer | Thought → Action → Observation cycle |
| **Tool Use?** | No tools, just reasoning | Uses external tools (e.g., APIs, databases) |
| **Example** | Solving a math problem | Fetching live weather data |
- **Zero-shot:** Model generates answers without examples.
- **Few-shot:** Model learns from a few provided examples before answering.

---

## **Actions: Enabling AI Agents to Engage with Their Environment**

### **1️⃣ Understanding Agent Actions**

- Actions are the concrete steps an **AI agent takes to interact with its environment**.(e.g. fetch data, execute commands).
- **Execution:** Actions can retrieve information, control devices, use tools, or communicate with users.
- **Examples:**
    - **Customer Service Agent:** Retrieves user data, provides support articles, escalates to human reps.
    - **Research Agent:** Searches the web, summarizes findings, cites sources.
    - **Automation Agent:** Executes commands, manipulates software interfaces.
    
    ---
    

### **2️⃣ Types of Agent Actions**

Different types of agents take actions differently 

| **Agent Type** | **Description** |
| --- | --- |
| **JSON Agent** | Outputs actions in structured JSON format. |
| **Code Agent** | Generates and executes code as an action. |
| **Function-Calling Agent** | Specialized JSON Agent designed to structure actions as function calls. |

Actions can serve many purposes:

| **Action Type** | **Description** |
| --- | --- |
| **Information Gathering** | Searching the web, querying databases. |
| **Tool Usage** | Calling APIs, performing calculations. |
| **Environment Interaction** | Controlling devices, automating tasks. |
| **Communication** | Engaging with users, coordinating multi-agent workflows. |

---

### **3️⃣ Stop and Parse Approach**

- **Ensures structured output:** The agent generates actions in a defined format (JSON or code).
- **Prevents unintended output:** The LLM/Agent STOPS token generation once the action is complete.
- **Enables external processing:** A parser extracts tool calls and parameters for execution.

**Example (JSON Agent Output for Weather Query):**

```json
Thought: I need to check the current weather for New York.
Action :
{
  "action": "get_weather",
  "action_input": {"location": "New York"}
}
```

- The agent stops once this action is formatted.
- An external system reads and executes the command.
- The LLM only handles text, and uses it to describe the action it wants to take and the parameters to supply to the tool.

---

### **4️⃣ Code Agents: Executing Actions as Code**

- **Concept:** Instead of outputting a JSON object, a Code Agent generates executable code (typically in high level lang -e.g.Python).

![image.png](attachment:db0114f0-beb1-4bce-81d0-20177a3d5069:image.png)

- **Advantages:**
    - **Expressiveness:** Handles complex logic (loops, conditionals).
    - **Modularity:** Code can be reused across tasks.
    - **Debugging:** Errors are easier to trace and fix.
    - **Direct Integration:** Can call APIs, perform calculations, process real-time data.

**Example (Code Agent for Weather Query):**

```python
def get_weather(city):
    import requests
    api_url = f"<https://api.weather.com/v1/location/{city}?apiKey=YOUR_API_KEY>"
    response = requests.get(api_url)
    if response.status_code == 200:
        return response.json().get("weather", "No weather information available")
    return "Error: Unable to fetch weather data."

result = get_weather("New York")
print(f"The current weather in New York is: {result}")
```

- The agent generates the function and executes it.
- Stops once the final response is printed.

---

### **Key Takeaways**

✔ **Actions enable AI agents to execute structured tasks** via JSON, function calls, or code.

✔ **The stop and parse approach ensures precise execution** and prevents unwanted output.

✔ **Code Agents offer greater flexibility** by generating dynamic scripts for complex workflows.

---

## **Observe: Capture + Integrate Feedback to Reflect and Adapt**

### **1️⃣ Role of Observations in AI Agents**

- Observations are **how an Agent perceives the consequences of its actions**.
- **Purpose:** Help agents evaluate the outcome of their actions.
- **Function:** They provide data on success, failure, or additional context for refinement.
- **Impact:** Ensures agents adapt dynamically based on real-world responses.
- The **iterative incorporation of feedback** ensures that the agent remains dynamically aligned with its goals

**Example:**

- Agent calls a weather API.
- Observation: `"Partly cloudy, 15°C, 60% humidity."`
- Agent appends this data and decides if more info is needed or if it's ready to respond.

---

### **2️⃣ Types of Observations**

| **Observation Type** | **Example** |
| --- | --- |
| **System Feedback** | Error messages, success confirmations, status codes. |
| **Data Changes** | Database updates, modified files, application states. |
| **Environmental Data** | Sensor readings, system metrics, resource usage. |
| **Response Analysis** | API responses, computation outputs, query results. |
| **Time-Based Events** | Deadline triggers, scheduled task completions. |

---

### **3️⃣ Observation Process: How Results Are Integrated**

1. **Parse Action:** Identify function calls and parameters.
2. **Execute Action:** Perform the task based on the parsed information.
3. **Append Observation:** Store the result as context for the next cycle.

**Example Workflow:**

- Agent sends a request to an API.
- API response (**observation**) is appended to the prompt.
- Agent analyzes feedback and refines the next step accordingly.

---

### **Key Takeaways**

✔ **Observations guide AI agents by integrating real-world feedback into their reasoning.**

✔ **They ensure continuous learning, refinement, and adaptive decision-making.**

✔ **Agents use observations to correct errors, verify success, and adjust future actions.**
