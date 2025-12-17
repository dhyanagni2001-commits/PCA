# PCA for Learning Word Embeddings

This project implements **Principal Component Analysis (PCA)** to learn **word embeddings** from a large word co-occurrence matrix derived from a Wikipedia corpus. The embeddings capture semantic and syntactic relationships between words and are evaluated on similarity, analogy, and bias analysis tasks.

---

## Project Overview
- Learns low-dimensional word embeddings using PCA
- Uses a co-occurrence matrix of the 3000 most frequent words
- Explores semantic structure, gender bias, and word relationships
- Evaluates embeddings using similarity and analogy tasks

---

## Implementation Details
All implementation is done in:
- `pca.py`
- `utils.py`

Only these files are modified and graded.

---

## Features Implemented
- PCA approximation for dimensionality reduction
- Word embedding construction
- Nearest-neighbor word similarity search
- Eigenvector interpretation
- Gender bias analysis via vector projections
- Word analogy solving
- Synonym vs antonym similarity evaluation

---

## How to Run
Run tasks incrementally using:
```bash
python3 pca_test.py <task_number>
