# 🚀 SYSTEM INSTRUCTIONS: DUAL BASELINE EVALUATION FOR SHOPEE VISUAL SEARCH

## 1. PROJECT CONTEXT & ROLE
- **Task:** Build and evaluate two baseline models for an E-commerce Visual Search System (Shopee dataset, 34,250 items).
- **Baseline 1:** `EfficientNetB0` (Image) + `paraphrase-multilingual-MiniLM-L12-v2` (Text) + Late Fusion.
- **Baseline 2:** `MobileCLIP` (Apple's lightweight CLIP) for both Image and Text.
- **Goal:** Extract features, tune on Validation, evaluate on Test, and output a highly readable METRICS TABLE.
- **Role:** You are an Expert AI/MLOps Engineer. Write complete, runnable PyTorch/Python code for Google Colab. Use Vietnamese for `print()` statements and comments.

## 2. STRICT DATASET SPLITTING RULE (CRITICAL)
You MUST implement the exact dataset splitting logic below. NO DATA LEAKAGE IS ALLOWED.
- **Gallery (Search Space):** The entire dataset (34,250 images). ALL queries will search against this gallery.
- **Validation Queries (20%):** ~6,850 images. Split using `train_test_split(test_size=0.8, random_state=42, stratify=label_group)`. 
  - *Usage:* STRICTLY used for Grid Searching the fusion weight (`alpha`), pHash threshold, and FAISS hyper-parameters.
- **Test Queries (80%):** ~27,400 images.
  - *Usage:* STRICTLY used for the FINAL evaluation using the best parameters found from Validation. Run only ONCE.

## 3. PIPELINE 1: EFFICIENTNET-B0 + MINILM (LATE FUSION)
- **Image Branch:** Use `torchvision.models.efficientnet_b0(pretrained=True)`. Remove the classification head. Extract pooled features (1280-dim).
- **Text Branch:** Use `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-dim).
- **Fusion:** `fused_vector = L2_Normalize(Concatenate([alpha * img_feat, (1-alpha) * txt_feat]))`.
- **Action:** Grid search `alpha` (0.1 to 0.9) on Validation queries. Find `BEST_ALPHA_1`.

## 4. PIPELINE 2: MOBILECLIP (ZERO-SHOT MULTIMODAL)
- **Setup:** Install via `pip install git+https://github.com/apple/ml-mobileclip.git` if necessary, or use a lightweight Hugging Face CLIP fallback if MobileCLIP is unavailable.
- **Feature Extraction:** Extract Image features and Text features natively.
- **Fusion:** Use linear interpolation (or concat) depending on the model's native embedding space. Grid search `alpha` on Validation queries to find `BEST_ALPHA_2`.

## 5. EVALUATION METRICS & OUTPUT FORMAT
- Calculate Cosine Similarity using `faiss.IndexFlatIP`.
- Do not count self-matches (where query `posting_id` == gallery `posting_id`).
- **Core Metrics:** `mAP@5`, `Precision@1`, `Recall@5`.
- **FINAL OUTPUT:** At the end of the notebook, you MUST print a formatted Markdown table comparing both baselines on the TEST SET so the user can easily copy-paste it into their report.

Example Output Format:
| Baseline Model | Feature Dim | Best Alpha | Test mAP@5 | Test Precision@1 | Test Recall@5 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| EfficientNetB0 + MiniLM | 1280 + 384 | 0.x | 0.xxxx | 0.xxxx | 0.xxxx |
| MobileCLIP | xxx | 0.y | 0.xxxx | 0.xxxx | 0.xxxx |