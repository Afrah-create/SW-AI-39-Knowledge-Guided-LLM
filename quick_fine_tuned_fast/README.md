---
library_name: transformers
tags:
- agricultural-ai
- llm
- fine-tuned
- crop-recommendation
- dialogpt
- text-generation
license: mit
datasets:
- ugandan-agricultural-data
- agricultural-literature
base_model: microsoft/DialoGPT-small
---

# Agricultural AI Fine-Tuned LLM

Fine-tuned DialoGPT-small model for agricultural crop recommendation and expert analysis generation.

## Model Overview

This model is a fine-tuned version of Microsoft's DialoGPT-small, trained specifically on agricultural domain knowledge for generating crop recommendations and expert analysis.

## Training Details

- **Base Model**: microsoft/DialoGPT-small
- **Training Data**: Agricultural knowledge graph triples, literature reviews, Ugandan agricultural dataset
- **Training Approach**: Domain-specific fine-tuning
- **Purpose**: Generate contextual agricultural recommendations and expert analysis

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Awongo/agricultural-llm-finetuned")
model = AutoModelForCausalLM.from_pretrained("Awongo/agricultural-llm-finetuned")

# Generate recommendation
prompt = "Given soil pH 6.2, organic matter 3%, and temperature 24°C, recommend suitable crops:"
inputs = tokenizer.encode(prompt, return_tensors="pt")
outputs = model.generate(inputs, max_length=200, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Integration

This model is used in the Agricultural AI system alongside:
- Graph Convolutional Networks (GCN) for entity embeddings
- Constraint-based reasoning engine
- Retrieval-Augmented Generation (RAG) pipeline

## Citation

```bibtex
@misc{agricultural-llm-finetuned,
  title={Agricultural AI Fine-Tuned Language Model for Crop Recommendation},
  year={2025},
  publisher={Hugging Face}
}
```
