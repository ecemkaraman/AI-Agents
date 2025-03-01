### **🖼️ Vision Agents with smolagents**

🔹 **Why Vision Agents?**

- **Beyond Text** → Process images, documents, web screenshots.
- **Real-World Use Cases** → ID verification, document processing, web navigation.
- **Agentic Approach** → Automate reasoning + visual analysis.

---

## **👀 Use Case: Guest Verification at Wayne Manor**

📌 **Scenario:**

Alfred needs to verify if **Wonder Woman** is actually **The Joker in disguise** before letting her into the party.

✅ **Solution:** A Vision Agent that:

- **Compares guest images** with a known dataset.
- **Uses a VLM (Vision-Language Model)** to analyze the visual data.
- **Verifies identity before granting access.**

---

## **📷 Step 1: Load Guest Images for Verification**

```python
from PIL import Image
import requests
from io import BytesIO

image_urls = [
    "<https://upload.wikimedia.org/wikipedia/commons/e/e8/The_Joker_at_Wax_Museum_Plus.jpg>",  # Joker image
    "<https://upload.wikimedia.org/wikipedia/en/9/98/Joker_%28DC_Comics_character%29.jpg>"  # Joker image
]

images = []
for url in image_urls:
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    images.append(image)

```

---

## **🛠 Step 2: Set Up a Vision Agent**

```python
from smolagents import CodeAgent, OpenAIServerModel

model = OpenAIServerModel(model_id="gpt-4o")

# Initialize agent
agent = CodeAgent(
    tools=[],
    model=model,
    max_steps=20,
    verbosity_level=2
)

response = agent.run(
    """
    Describe the costume and makeup of the character in these photos.
    Identify whether the guest is The Joker or Wonder Woman.
    """,
    images=images
)

```

📌 **Sample Output:**

```
{
    'Costume - First Image': 'Purple coat, mustard-yellow shirt, white face paint, exaggerated smile.',
    'Costume - Second Image': 'Dark suit, flower on lapel, green hair, red lips.',
    'Character Identity': 'This matches known depictions of The Joker.'
}

```

✅ **Alfred successfully blocks The Joker from sneaking in!**

---

## **🌐 Step 3: Dynamic Image Retrieval (For Guests Not in Database)**

🔹 **Problem:** If a guest is **not** in the dataset, Alfred needs **real-time** image search.

🔹 **Solution:** Use **web browsing automation** to find images & verify identity.

---

## **🔍 Step 4: Install Dependencies**

```bash
pip install "smolagents[all]" helium selenium python-dotenv

```

---

## **🌎 Step 5: Create Web Browsing Tools**

```python
from selenium.webdriver.common.keys import Keys
import helium
from smolagents import tool

def search_item_ctrl_f(text: str, nth_result: int = 1) -> str:
    """Search for text on a page using Ctrl + F."""
    elements = driver.find_elements(By.XPATH, f"//*[contains(text(), '{text}')]")
    if nth_result > len(elements):
        raise Exception(f"Match {nth_result} not found (only {len(elements)} matches)")
    elem = elements[nth_result - 1]
    driver.execute_script("arguments[0].scrollIntoView(true);", elem)
    return f"Found {len(elements)} matches for '{text}'. Focused on match {nth_result}."

def go_back() -> None:
    """Go back to the previous page."""
    driver.back()

def close_popups() -> str:
    """Closes popups using the Escape key."""
    webdriver.ActionChains(driver).send_keys(Keys.ESCAPE).perform()
    return "Pop-ups closed."

```

---

## **📸 Step 6: Capture Web Screenshots for Agent Memory**

```python
from smolagents.core import ActionStep
from time import sleep
from PIL import Image
import helium

def save_screenshot(step_log: ActionStep, agent: CodeAgent) -> None:
    """Captures a screenshot and stores it in agent memory."""
    sleep(1.0)  # Allow animations to complete
    driver = helium.get_driver()
    png_bytes = driver.get_screenshot_as_png()
    image = Image.open(BytesIO(png_bytes))
    print(f"Captured screenshot: {image.size} pixels")

    step_log.observations_images = [image.copy()]
    step_log.observations = f"Current URL: {driver.current_url}"

```

---

## **🤖 Step 7: Build the Web Browsing Vision Agent**

```python
from smolagents import CodeAgent, OpenAIServerModel, DuckDuckGoSearchTool

model = OpenAIServerModel(model_id="gpt-4o")

agent = CodeAgent(
    tools=[DuckDuckGoSearchTool(), go_back, close_popups, search_item_ctrl_f],
    model=model,
    additional_authorized_imports=["helium"],
    step_callbacks=[save_screenshot],
    max_steps=20,
    verbosity_level=2,
)

```

---

## **🚀 Step 8: Run the Vision Agent for Real-Time Guest Verification**

```python
agent.run("""
I am Alfred, the butler of Wayne Manor, responsible for verifying guest identities. A superhero claims to be Wonder Woman, but I must confirm this.

Search for images of Wonder Woman and describe her appearance in detail.
Then, visit Wikipedia and gather key details about her outfit and features.
Use this information to verify the guest's identity before granting access.
""")

```

📌 **Final Output:**

```
Wonder Woman is depicted wearing a red and gold bustier, blue shorts/skirt with white stars, a golden tiara, silver bracelets, and a golden Lasso of Truth.
She is Princess Diana of Themyscira, also known as Diana Prince.
✅ Identity Verified! Guest is Wonder Woman.

```

---

## **🎯 Key Takeaways**

✅ **Static Image Verification** → Use a dataset of known guests.

✅ **Dynamic Web Search** → Browse for visual details if unknown.

✅ **Automated Screenshot Capture** → Store and process web images dynamically.

✅ **Multi-Step Reasoning** → Combine text & vision for robust verification.

🚀 **Next Steps: Scaling Vision Agents for Document Processing & Object Recognition!**
