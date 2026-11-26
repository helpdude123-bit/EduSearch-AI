# EduSearch AI

## Smart Educational Document Search Engine

Discover Knowledge. Search Smarter.

### NIELIT AIDA - NLP Practical Project

Problem 5: Educational Document Search Engine

### Project Overview

EduSearch AI retrieves educational documents
using TF-IDF vectorization and cosine similarity.

### Features

- 30 educational documents
- 6 subject categories
- Natural language search
- Top 5 ranked document results
- Configurable similarity threshold
- Category-based filtering
- Full document library
- Technical evaluation dashboard
- No-match and invalid-input handling

### Technical Configuration

- TF-IDF vocabulary: 2119 features
- Matrix dimensions: (30, 2119)
- Non-zero values: 2616
- Unigrams and bigrams
- English stop-word removal
- Cosine similarity-based ranking

### Testing Results

- Total test cases: 10
- Passed: 9
- Partial: 1
- Failed: 0

These are project-specific test results,
not a general retrieval accuracy benchmark.

### Dataset Source

Original AI-assisted educational notes
prepared specifically for this project.

### Installation

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

### Limitations

- Dependence on overlapping vocabulary
- Limited understanding of synonyms
- Limited semantic context recognition
- Similarity does not guarantee relevance

### Academic Purpose

Developed as an individual NLP practical project
for NIELIT AIDA.
