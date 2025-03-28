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
  {
    "name": "Card",
    "description": "A container for content that groups related information in a flexible and extensible content container.",
    "code": "<Card>\n  <CardHeader>Card Title</CardHeader>\n  <CardBody>This is some text within a card body.</CardBody>\n</Card>",
    "tags": ["card", "container", "UI component"],
    "source": "ShadCN"
  },
  {
    "name": "Checkbox",
    "description": "A square box that can be checked or unchecked to select or deselect an option.",
    "code": "<Checkbox label=\"Accept terms and conditions\" />",
    "tags": ["checkbox", "form", "UI component"],
    "source": "ShadCN"
  },
  {
    "name": "Dialog",
    "description": "A window overlaid on either the primary window or another dialog window, rendering the content underneath inert.",
    "code": "<Dialog>\n  <DialogTrigger><Button>Open Dialog</Button></DialogTrigger>\n  <DialogContent>\n    <DialogTitle>Dialog Title</DialogTitle>\n    <DialogDescription>This is the dialog content.</DialogDescription>\n  </DialogContent>\n</Dialog>",
    "tags": ["dialog", "modal", "UI component"],
    "source": "ShadCN"
  },
  {
    "name": "Dropdown Menu",
    "description": "A toggleable menu that allows the user to choose one value from a predefined list.",
    "code": "<DropdownMenu>\n  <DropdownMenuTrigger><Button>Options</Button></DropdownMenuTrigger>\n  <DropdownMenuContent>\n    <DropdownMenuItem>Item 1</DropdownMenuItem>\n    <DropdownMenuItem>Item 2</DropdownMenuItem>\n  </DropdownMenuContent>\n</DropdownMenu>",
    "tags": ["dropdown", "menu", "UI component"],
    "source": "ShadCN"
  },
  {
    "name": "Input",
    "description": "A field that allows users to enter data.",
    "code": "<Input type=\"text\" placeholder=\"Enter your name\" />",
    "tags": ["input", "form", "UI component"],
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

