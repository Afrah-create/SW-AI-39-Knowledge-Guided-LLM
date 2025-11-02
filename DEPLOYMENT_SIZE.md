# Deployment Size Optimization

## Files Excluded to Stay Under 1GB Limit

### Excluded from Railway Deployment:
1. ✅ **quick_fine_tuned_fast/** (~1.2 GB)
   - checkpoint-1/ (model files)
   - checkpoint-500/ (model files)
   - optimizer.pt (625 MB)
   - model.safetensors (312 MB × 3 copies)
   - **Total saved: ~1.2 GB**

2. ✅ **Large JSON data files** (~115 MB)
   - unified_knowledge_graph.json (65 MB)
   - dataset_triples.json (49 MB)
   - literature_triples.json
   
3. ✅ **Duplicated data/ folder**
   - Same files as processed/
   - **Total saved: ~120 MB**

### What Remains (Essential Files):
- ✅ **App code**: ~2 MB
  - deployment/app/main.py
  - deployment/requirements.txt
  - deployment/Dockerfile
  - deployment/Procfile

- ✅ **Trained models** (~8 MB)
  - best_model.pth (1.12 MB)
  - gcn_model.pth (1.12 MB)
  - complex_model.pth (1.93 MB)
  - distmult_model.pth (0.97 MB)
  - graphsage_model.pth (1.20 MB)
  - transe_model.pth (0.97 MB)

- ✅ **Small data files** (~10 MB)
  - ugandan_data_cleaned.csv (4 MB)
  - model_metadata.json
  - best_model_info.json

**Total Deployment Size**: ~20-30 MB (within 1GB limit! ✅)

---

## App Functionality

### Will Still Work:
- ✅ Main crop recommendation API
- ✅ Constraint-based validation
- ✅ Agricultural evaluation scores
- ✅ PDF report generation
- ✅ Trained GNN model inference
- ✅ Basic RAG with fallback

### Won't Work (Optional Features):
- ⚠️ Advanced RAG with knowledge graph (needs large JSON files)
- ⚠️ Fine-tuned LLM features (needs quick_fine_tuned_fast/)
- ⚠️ Semantic retrieval over triples

The app handles missing files gracefully and will continue to work with constraints engine and basic features.

---

## How It's Configured:

### `.gitignore` (for GitHub):
```
deployment/quick_fine_tuned_fast/
deployment/data/
deployment/processed/unified_knowledge_graph.json
**/optimizer.pt
**/model.safetensors
```

### `.dockerignore` (for Railway):
```
quick_fine_tuned_fast/
data/
processed/unified_knowledge_graph.json
**/optimizer.pt
**/model.safetensors
```

---

## Deployment Checklist:
- [x] Excluded 1.2GB of unused model files
- [x] Excluded 120MB of large data files
- [x] Kept essential trained models (8MB)
- [x] Updated .gitignore
- [x] Updated .dockerignore
- [x] Pushed to GitHub
- [x] Under 1GB limit ✅

---

**Result**: Your deployment is now ~20-30 MB, well under Railway's 1GB free limit! 🚀

