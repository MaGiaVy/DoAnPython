# 🚀 SYSTEM INSTRUCTIONS FOR AI: MULTIMODAL VISUAL SEARCH (SHOPEE)

## 1. PROJECT CONTEXT & ROLE
- **Task:** Build a high-performance Visual Search & Instance Retrieval System for a Shopee E-commerce dataset (34,250 products).
- **Core Architecture:** `CLIP (Image Branch)` + `TF-IDF (Text Branch)` + `Late Fusion (Concat)` + `pHash Reranking`.
- **Target Metric:** Maximize `mAP@5` (Must strictly exceed 0.80). Secondary metrics: `Precision@1`, `Recall@5`.
- **Role:** You are an Expert Computer Vision & MLOps Engineer. Write highly optimized, bug-free PyTorch & Scikit-learn code for Google Colab (T4 GPU - 15GB VRAM). Code comments must be in Vietnamese.

## 2. STRICT WORKFLOW & EVALUATION CONSTRAINTS
- **No Data Leakage:** You MUST split the dataset into Validation (20%) and Test (80%) using `stratify=label_group`.
- **Validation Set (20%):** Strictly used ONLY to Grid Search the optimal `alpha` weight.
- **Test Set (80%):** Strictly used ONLY for the FINAL ONE-TIME evaluation with Post-processing.

## 3. PIPELINE SPECIFICATIONS

### STEP 1: Image Feature Extraction (CLIP)
- Use model: `openai/clip-vit-base-patch32` (via Hugging Face `transformers`).
- Extract ONLY the image features (`image_embeds`). DO NOT use CLIP's text encoder.
- Apply L2 Normalization immediately. Save to `processed/clip_img.npy`.

### STEP 2: Text Feature Extraction (TF-IDF + SVD)
- Target column: `title`. Clean basic text (lowercase, remove special chars).
- Initialize `TfidfVectorizer` with anti-typo settings: 
  - `analyzer='char_wb'`, `ngram_range=(3, 5)`, `max_features=30000`, `sublinear_tf=True`.
- Reduce dimensions using `TruncatedSVD(n_components=256)`.
- Apply L2 Normalization immediately. Save to `processed/tfidf_txt.npy`.

### STEP 3: Grid Search Alpha (Late Fusion on Validation Set)
- Feature Fusion Formula: 
  `fused_vector = L2_Normalize( Concatenate( [alpha * img_features, (1 - alpha) * txt_features], axis=1 ) )`
- Use `faiss.IndexFlatIP` to calculate Cosine Similarity.
- Loop `alpha` from `0.1` to `0.9` (step 0.1). Calculate `val_mAP@5` for each.
- Output the `BEST_ALPHA`.

### STEP 4: pHash Boosting & Reranking (Final Test Set Evaluation)
- Apply `BEST_ALPHA` to create the fused test features.
- Query against the gallery using FAISS to retrieve the **Top 50** candidates (`top_scores`, `top_indices`).
- **Reranking Logic:**
  - Loop through each candidate in the Top 50.
  - Calculate Hamming Distance between query's `image_phash` and candidate's `image_phash`.
  - **Boost:** If `Hamming Distance == 0`, add `+1.0` to the FAISS cosine score.
  - Re-sort candidates by the new boosted scores (Descending).
  - Slice the Final **Top 5** candidates.
- Compute final `mAP@5`, `Precision@1`, `Recall@5` and print nicely formatted results.
- Export results to `results/final_metrics_clip_tfidf.csv`.