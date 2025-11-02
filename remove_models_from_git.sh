#!/bin/bash
# Script to remove model files from Git LFS and Git tracking
# Run this AFTER pushing to remove models from future commits

echo "Removing model files from Git LFS tracking..."

# Remove from Git LFS
git lfs untrack "models/*.pth"
git lfs untrack "processed/trained_models/*.pth"
git lfs untrack "quick_fine_tuned_fast/**/*.safetensors"
git lfs untrack "quick_fine_tuned_fast/**/*.pth"
git lfs untrack "quick_fine_tuned_fast/**/*.pt"
git lfs untrack "quick_fine_tuned_fast/**/*.bin"

# Remove from Git index (but keep files locally)
git rm --cached -r models/
git rm --cached -r quick_fine_tuned_fast/
git rm --cached -r processed/trained_models/

# Commit the removal
git commit -m "Remove model files from Git tracking (now on Hugging Face)"

echo "✅ Model files removed from Git tracking"
echo "⚠️  Note: Files still exist locally but won't be tracked by Git"
echo "📤 Next: Push this commit: git push origin main"

