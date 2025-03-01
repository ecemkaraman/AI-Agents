### **🤖 Multi-Agent Systems in AI Agents**

🔹 **Why Multi-Agent Systems?**

- **Collaboration** → Specialized agents handle different tasks.
- **Modularity** → Easier to manage and scale.
- **Efficiency** → Reduces token usage, improving speed & cost.

✅ **Example: A Multi-Agent RAG System**

- **Web Agent** → Searches the internet.
- **Retriever Agent** → Fetches from knowledge bases.
- **Image Gen Agent** → Creates visuals.
- **Manager Agent** → Orchestrates everything.

---

### **🚀 Multi-Agent Systems in Action**

💡 **Scenario:** Alfred needs to find **Batman filming locations** worldwide and calculate travel time for cargo planes.

### **🔧 Step 1: Install Required Packages**

```bash
pip install 'smolagents[litellm]' matplotlib geopandas shapely kaleido -q

```

---

## **🛠 Step 2: Create a Tool to Calculate Cargo Plane Travel Time**

```python
import math
from typing import Optional, Tuple
from smolagents import tool

def calculate_cargo_travel_time(
    origin_coords: Tuple[float, float], destination_coords: Tuple[float, float], cruising_speed_kmh: Optional[float] = 750.0
) -> float:
    """Calculates travel time for a cargo plane using great-circle distance."""
    def to_radians(degrees: float) -> float:
        return degrees * (math.pi / 180)

    lat1, lon1 = map(to_radians, origin_coords)
    lat2, lon2 = map(to_radians, destination_coords)

    EARTH_RADIUS_KM = 6371.0
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    distance = EARTH_RADIUS_KM * c * 1.1  # Account for non-direct routes
    return round((distance / cruising_speed_kmh) + 1.0, 2)  # Add 1hr for takeoff/landing

print(calculate_cargo_travel_time((41.8781, -87.6298), (-33.8688, 151.2093)))

```

---

## **🔍 Step 3: Set Up a Simple Agent for Information Retrieval**

📌 **Task:** Find all **Batman filming locations & supercar factories** and **calculate travel time to Gotham.**

```python
import os
from smolagents import CodeAgent, GoogleSearchTool, HfApiModel, VisitWebpageTool

model = HfApiModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct", provider="together")

agent = CodeAgent(
    model=model,
    tools=[GoogleSearchTool("serper"), VisitWebpageTool(), calculate_cargo_travel_time],
    additional_authorized_imports=["pandas"],
    max_steps=20,
)

task = """Find all Batman filming locations in the world, calculate the time to transfer via cargo plane to Gotham (40.7128° N, 74.0060° W), and return them as a pandas dataframe. Also list supercar factories with the same cargo transfer time."""
result = agent.run(task)
print(result)

```

📌 **Sample Output:**

```
| Location                         | Travel Time (hours) |
|----------------------------------|---------------------|
| Glasgow, UK                      | 8.60               |
| London, UK                        | 9.17               |
| Wall Street, NYC, USA             | 1.00               |
| Hong Kong, China                  | 19.99              |
| McLaren Factory, UK               | 9.13               |

```

---

## **✌️ Step 4: Improving Efficiency with a Multi-Agent Approach**

🔹 **Why split tasks?**

- **More focus per agent** → Each agent specializes.
- **Lower token costs** → Reduces memory overload.
- **Better performance** → Faster and more accurate results.

📌 **New Structure:**

1️⃣ **Web Agent** → Searches for locations & factories.

2️⃣ **Manager Agent** → Handles planning & data visualization.

---

## **🌐 Step 5: Setting Up a Dedicated Web Agent**

```python
web_agent = CodeAgent(
    model=model,
    tools=[
        GoogleSearchTool(provider="serper"),
        VisitWebpageTool(),
        calculate_cargo_travel_time,
    ],
    name="web_agent",
    description="Searches the web for information",
    verbosity_level=0,
    max_steps=10,
)

```

---

## **🧠 Step 6: Creating a Manager Agent with Planning & Visualization**

```python
from smolagents.utils import encode_image_base64, make_image_url
from smolagents import OpenAIServerModel
from PIL import Image

def check_reasoning_and_plot(final_answer, agent_memory):
    multimodal_model = OpenAIServerModel("gpt-4o", max_tokens=8096)
    filepath = "saved_map.png"
    image = Image.open(filepath)

    prompt = f"Here are agent steps: {agent_memory.get_succinct_steps()}. Verify the plot for accuracy."

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": make_image_url(encode_image_base64(image))}},
            ],
        }
    ]

    output = multimodal_model(messages).content
    print("Feedback: ", output)
    if "FAIL" in output:
        raise Exception(output)
    return True

```

📌 **Assigning Tasks to Manager Agent**

```python
manager_agent = CodeAgent(
    model=HfApiModel("deepseek-ai/DeepSeek-R1", provider="together", max_tokens=8096),
    tools=[calculate_cargo_travel_time],
    managed_agents=[web_agent],
    additional_authorized_imports=["geopandas", "plotly", "shapely", "json", "pandas", "numpy"],
    planning_interval=5,
    verbosity_level=2,
    final_answer_checks=[check_reasoning_and_plot],
    max_steps=15,
)

```

---

## **🔍 Step 7: Visualizing Multi-Agent Architecture**

```python
manager_agent.visualize()

```

📌 **Outputs a Graph Representation:**

```
CodeAgent | deepseek-ai/DeepSeek-R1
├── ✅ Imports: ['geopandas', 'plotly', 'shapely', 'json', 'pandas', 'numpy']
├── 🛠 Tools:
│   ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
│   ┃ Name                        ┃ Description                   ┃
│   ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│   │ calculate_cargo_travel_time │ Computes cargo flight time.   │
│   └─────────────────────────────┴───────────────────────────────┘
└── 🤖 Managed Agents:
    └── web_agent | CodeAgent | Qwen/Qwen2.5-Coder-32B-Instruct
        ├── 📝 Description: Searches web for information
        └── 🛠️ Tools:
            ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
            ┃ Name                        ┃ Description                   ┃
            ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
            │ web_search                  │ Performs Google search.       │
            │ visit_webpage               │ Reads webpage content.        │
            └─────────────────────────────┴───────────────────────────────┘

```

---

## **📊 Step 8: Running the Multi-Agent System**

```python
manager_agent.run("""
Find all Batman filming locations, calculate the time to transfer via cargo plane to Gotham.
Also find supercar factories with the same transfer time. Plot the data as a scatter map and save as saved_map.png.
""")

```

📌 **Final Output:**

✅ **Map of filming locations & factories, color-coded by travel time!**

---

## **🎯 Key Takeaways**

✅ **Multi-Agent Systems improve performance by distributing tasks.**

✅ **Web agents handle searches, manager agents process & visualize data.**

✅ **Splitting memory between agents reduces token cost & improves speed.**

✅ **Visualization tools (geopandas, plotly) enhance data representation.**

**🚀 Next: Deploying & Scaling Multi-Agent Workflows!**
