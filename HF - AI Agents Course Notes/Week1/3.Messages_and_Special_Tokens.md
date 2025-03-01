### **Messages and Special Tokens**

### **1. Chat Messages vs. Prompts**

- **User Interaction**: Users interact with LLMs through chat messages, not single prompt sequences.
- **Model Processing**: Behind the scenes, messages are concatenated into a single prompt before being fed into the model.
- **No Memory**: LLMs do not "remember" past interactions; every response is based on the full input prompt.

---

### **2. Special Tokens and Message Formatting**

- **Special Tokens**: Define start and end points for messages (e.g. `<|im_start|>`, `<|im_end|>`)
- **Message Types**:
    - **System Messages**: Define model behavior →persistent instructions guiding every subsequent interaction (e.g., "You are a polite customer service agent.").
        - In agents, system messages also give info on available tools, how to format actions, the thought process etc.
    - **User Messages**: Input from the user.
    - **Assistant Messages**: Model-generated responses.
- **Example of Formatted Prompt** (SmolLM2 chat template):
    
    ```
    <|im_start|>system
    You are a helpful AI assistant.<|im_end|>
    <|im_start|>user
    I need help with my order.<|im_end|>
    <|im_start|>assistant
    I'd be happy to help. Could you provide your order number?<|im_end|>
    ```
    

---

### **3. Chat Templates**

- **Purpose**: Structure conversations between language models and users.
- Convert structured chat messages into a single prompt format for the model and ensure correct token formatting.
- **Role in Multi-Turn Conversations**: Helps maintain conversation history/context for coherence.
- **Different Templates for Different Models**: Each model (GPT-4, Llama 3, DeepSeek-R1) has unique special tokens (e.g. EOS) and formatting rules/delimiters.

---

### **4. Base Models vs. Instruct Models**

- **Base Model**:
    - Trained on raw text to predict the next token.
    - Example: `SmolLM2-135M`.
- **Instruct Model**:
    - Fine-tuned to follow instructions and maintain conversation structure.
    - Example: `SmolLM2-135M-Instruct`.
- **Chat Templates for Instruct Models**: Ensure correct+consistent formatting of prompts to convert a base model→instruct model
- e.g. **ChatML** = standardized format for structuring multi-turn conversations in LLMs using role-based tags (system, user, assistant) for clear parsing.
- A base model can be fine-tuned on different chat templates
- Each instruct model has unique formats and tokens; chat templates standardize prompt formatting.

---

### **5. Chat Template Implementation**

- **Jinja2-Based Templates**: Used in Transformers to convert messages into structured input.
    - Templating system for dynamically formatting chat prompts using Jinja2 syntax, ensuring correct structure for different LLMs.
    - Example:
        
        ```python
        {% for message in messages %}
        <|im_start|>{{ message['role'] }}
        {{ message['content'] }}<|im_end|>
        {% endfor %}
        ```
        
- **Example Output for Messages:**
    
    ```python
    <|im_start|>system
    You are a technical assistant.<|im_end|>
    <|im_start|>user
    Can you explain chat templates?<|im_end|>
    <|im_start|>assistant
    A chat template structures user-AI interactions...<|im_end|>
    ```
    
    - Transformers library auto-formats chat templates during tokenization.

---

### **6. Tokenizer and Chat Templates in Transformers**

- The easiest way to ensure your LLM receives a conversation correctly formatted is to use the `chat_template` from the model’s tokenizer.

```python
messages = [
    {"role": "system", "content": "You are an AI assistant with access to various tools."},
    {"role": "user", "content": "Hi !"},
    {"role": "assistant", "content": "Hi human, what can help you with ?"},
]
```

- To convert this conversation into a prompt, load the tokenizer and call **`apply_chat_template()`**:
    
    ```python
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")
    **rendered_prompt** = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ```
    
- *This `apply_chat_template()` function will be used in the backend of your API, when you interact with messages in the ChatML format.*
- **Automatically Formats Messages**: Ensures model receives properly structured inputs.
- **Key Takeaway**: Chat templates and tokenization standardize user-AI interaction across models.

---
