---
library_name: pytorch
tags:
- graph-neural-network
- knowledge-graph
- agricultural-ai
- crop-recommendation
- gcn
- graph-embeddings
license: mit
datasets:
- ugandan-agricultural-data
---

# Agricultural AI Graph Embedding Models

Graph neural network models trained on Ugandan agricultural knowledge graph for crop recommendation.

## Model Overview

This repository contains multiple graph embedding models trained on an agricultural knowledge graph with 175,318 triples representing crop-soil-climate relationships.

## Models Included

### Best Model: GCN (Graph Convolutional Network)
- **File**: `best_model.pth`
- **Accuracy**: 87.28%
- **F1-Score**: 85.71%
- **ROC-AUC**: 96.90%
- **Embedding Dimension**: 100
- **Entities**: 2,513
- **Relations**: 15

### Individual Models
1. **GCN Model** (`gcn_model.pth`) - Best performing
2. **TransE Model** (`transe_model.pth`) - Translation-based
3. **DistMult Model** (`distmult_model.pth`) - Bilinear
4. **ComplEx Model** (`complex_model.pth`) - Complex embeddings
5. **GraphSAGE Model** (`graphsage_model.pth`) - Sampling-based

## Model Metadata

The `model_metadata.json` file contains:
- Entity to ID mappings (2,513 entities)
- Relation to ID mappings (15 relations)
- ID to entity mappings
- Model configuration parameters

## Usage

```python
import torch
from huggingface_hub import hf_hub_download

# Download model
model_path = hf_hub_download(
    repo_id="Awongo/soil-crop-recommendation-model",
    filename="best_model.pth"
)

# Download metadata
metadata_path = hf_hub_download(
    repo_id="Awongo/soil-crop-recommendation-model",
    filename="model_metadata.json"
)

# Load model (pseudo-code - adjust to your model architecture)
# model = GCNModel(num_entities=2513, num_relations=15, embedding_dim=100)
# model.load_state_dict(torch.load(model_path, map_location='cpu'))
# model.eval()
```

## Training Data

- **Knowledge Graph**: 175,318 triples
- **Dataset**: Ugandan agricultural data
- **Literature**: 52 research papers
- **Crops**: 8 major crops (maize, rice, beans, cassava, sweet potato, banana, coffee, cotton)

## Application

Used in production for agricultural crop recommendations based on:
- Soil properties (pH, organic matter, nutrients)
- Climate conditions (temperature, rainfall)
- Knowledge graph embeddings

## Citation

```bibtex
@misc{agricultural-ai-graph-models,
  title={Agricultural AI Graph Embedding Models for Crop Recommendation},
  year={2025},
  publisher={Hugging Face}
}
```
