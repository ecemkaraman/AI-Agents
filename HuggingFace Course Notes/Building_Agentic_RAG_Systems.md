### **🔍 Building Agentic RAG Systems**

💡 **Agentic RAG (Retrieval-Augmented Generation)** → Extends traditional RAG by adding **intelligent control over retrieval & generation** for better accuracy.

🔹 **Traditional RAG Limitations**

- Single retrieval step → May miss relevant data.
- Direct semantic search → Can overlook key insights.

✅ **Agentic RAG Advantages**

- Dynamically **formulates search queries**.
- **Critiques retrieved results** & refines queries.
- Conducts **multi-step retrieval** for deeper insights.

---

## **🔹 Basic Retrieval with DuckDuckGo**

🎯 **Goal:** Search the web for luxury superhero-themed party ideas.

📌 **Example: Alfred's Party Planner Agent**

```python
from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel

# Initialize search tool & model
search_tool = DuckDuckGoSearchTool()
model = HfApiModel()

agent = CodeAgent(model=model, tools=[search_tool])

# Run search query
response = agent.run("Search for luxury superhero-themed party ideas, including decorations, entertainment, and catering.")
print(response)

```

### **🚀 Process Overview**

1️⃣ **Analyzes Query** → Identifies key elements (luxury, superhero, party).

2️⃣ **Retrieves Data** → Uses DuckDuckGo to fetch the most relevant insights.

3️⃣ **Synthesizes Information** → Converts results into an actionable event plan.

4️⃣ **Stores for Future Use** → Saves relevant insights for subsequent tasks.

---

## **📚 Custom Knowledge Base Tool**

🎯 **Goal:** Use a **vector database** for **faster & more accurate** knowledge retrieval.

✅ **Why?**

- 🧠 **Retains domain-specific knowledge**.
- 🔍 **Performs efficient semantic search**.
- 🏗 **Combines structured & retrieved data**.

📌 **Example: Party Planning Knowledge Base**

```python
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from smolagents import Tool, CodeAgent, HfApiModel
from langchain_community.retrievers import BM25Retriever

class PartyPlanningRetrieverTool(Tool):
    name = "party_planning_retriever"
    description = "Retrieves luxury superhero-themed party planning ideas."

    inputs = {"query": {"type": "string", "description": "Party planning search query."}}
    output_type = "string"

    def __init__(self, docs, **kwargs):
        super().__init__(**kwargs)
        self.retriever = BM25Retriever.from_documents(docs, k=5)

    def forward(self, query: str) -> str:
        docs = self.retriever.invoke(query)
        return "\\nRetrieved ideas:\\n" + "".join(
            [f"\\n\\n===== Idea {i} =====\\n" + doc.page_content for i, doc in enumerate(docs)]
        )

# Simulated party planning database
party_ideas = [
    {"text": "A luxury superhero-themed masquerade ball with gold decor.", "source": "Party Ideas"},
    {"text": "Live DJ playing themed music for superheroes.", "source": "Entertainment Ideas"},
    {"text": "Catering with superhero-inspired dishes like 'Iron Man’s Power Steak'.", "source": "Catering Ideas"},
    {"text": "Gotham skyline projections for event ambiance.", "source": "Decoration Ideas"},
    {"text": "VR superhero simulations for guests.", "source": "Entertainment Ideas"},
]

# Convert to document format
source_docs = [Document(page_content=doc["text"], metadata={"source": doc["source"]}) for doc in party_ideas]

# Split documents for retrieval
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs_processed = text_splitter.split_documents(source_docs)

# Create retriever tool
party_planning_retriever = PartyPlanningRetrieverTool(docs_processed)

# Initialize agent with knowledge base tool
agent = CodeAgent(tools=[party_planning_retriever], model=HfApiModel())

# Query agent for luxury superhero party ideas
response = agent.run("Find luxury superhero-themed party ideas.")
print(response)

```

### **🔍 How It Works**

1️⃣ **Checks Documentation** → Searches party knowledge base first.

2️⃣ **Synthesizes Insights** → Combines retrieved knowledge with user query.

3️⃣ **Stores & Contextualizes** → Saves relevant results for efficient re-use.

---

## **🧠 Advanced Retrieval Capabilities**

### **🛠️ Key Features for Optimized RAG**

✅ **Query Reformulation** → Enhances search terms for **better retrieval**.

✅ **Multi-Step Retrieval** → Uses **initial results to refine subsequent searches**.

✅ **Source Integration** → Merges insights from **multiple sources** (web + knowledge base).

✅ **Result Validation** → Filters **irrelevant or incorrect** responses before final output.

### **⚡ Optimizing Agentic RAG**

📌 **Choosing the Right Tools** → Selects retrieval methods based on context.

📌 **Memory Management** → Avoids redundant searches by storing insights.

📌 **Fallback Strategies** → Ensures useful output **even when retrieval fails**.

📌 **Validation Steps** → Confirms **accuracy & relevance** before final response.

---

### **🚀 Key Takeaways**

✅ **Agentic RAG enables dynamic, multi-step, and intelligent retrieval.**

✅ **Combines traditional web search with knowledge bases for precision.**

✅ **Optimizes search queries, integrates multiple sources & refines results.**

✅ **Validates & stores data for context-aware, efficient responses.**

🔗 **Next: Implementing memory & improving retrieval accuracy for even better AI Agents!**
