# **Build a UI Component Retrieval System with Pinecone (RAG-Based Semantic Search)**

## Overview

This project demonstrates how to build a UI component retrieval system using Pinecone and a RAG (Retrieval-Augmented Generation) approach for semantic search. The system allows users to query for UI components based on natural language descriptions and retrieves relevant components from a pre-defined dataset.

## Initial Setup

To get started, open Google Colab and set up a new empty notebook.

### Step 1: Install Dependencies

Run the following commands to install the required libraries:

```bash
pip install pinecone
pip install sentence-transformers
```


```bash
from sentence_transformers import SentenceTransformer
import json

# Load the open-source embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Example UI component dataset
components = [
{
    "name": "Accordion",
    "description": "A vertically stacked set of interactive headings that each reveal or hide associated sections of content.",
    "code": "<Accordion>\n  <AccordionItem>\n    <AccordionHeader>Section 1</AccordionHeader>\n    <AccordionPanel>Content for section 1.</AccordionPanel>\n  </AccordionItem>\n  <AccordionItem>\n    <AccordionHeader>Section 2</AccordionHeader>\n    <AccordionPanel>Content for section 2.</AccordionPanel>\n  </AccordionItem>\n</Accordion>",
    "tags": ["accordion", "collapse", "UI component"],
    "source": "ShadCN"
  },
  {
    "name": "Alert",
    "description": "A banner that displays important, succinct information, typically at the top of the page.",
    "code": "<Alert>\n  <AlertTitle>Success!</AlertTitle>\n  <AlertDescription>Your operation was completed successfully.</AlertDescription>\n</Alert>",
    "tags": ["alert", "notification", "UI component"],
    "source": "ShadCN"
  },
  {
    "name": "Button",
    "description": "An interactive element that triggers an action or event when clicked.",
    "code": "<Button variant=\"primary\">Click Me</Button>",
    "tags": ["button", "interactive", "UI component"],
    "source": "ShadCN"
  },
]
 #add more entries like the ones mentioned above

# Generate embeddings for each component
for component in components:
    text_to_embed = f"{component['name']} {component['description']} {', '.join(component['tags'])} {component['code']}"
    component["embedding"] = model.encode(text_to_embed).tolist()

# Save to JSON
with open("ui_components_with_embeddings.json", "w") as f:
    json.dump(components, f, indent=4)

print("✅ Embeddings generated and saved!")

