# Search Engine "from Scratch"
A search engine built "from scratch" in Python for Information Retrieval course assignment.

## Requirements
```bash
pip install nltk tqdm
```

NLTK stopwords are downloaded automatically on first run (internet connection required).

## How to Run
**1. Build the index**
```bash
python bsbi.py
```

**2. Run sample queries, SPIMI + WAND verification, and timing comparison**
```bash
python search.py
```

**3. Evaluate system quality**
```bash
python evaluation.py
```
> **Note:** ALWAYS re-run `bsbi.py` IF the collection changes.

## Features

### 1. Index Compression

Two compression schemes:

| Class | Level | Method |
|---|---|---|
| `VBEPostings` | Byte | Variable-Byte Encoding on docID gaps |
| `EliasGammaPostings` | Bit | Elias-Gamma encoding on docID gaps, zero-padded to byte boundary |

Both encode **docID gaps** for compression. TF lists are encoded as absolute values.

### 2. BM25 Retrieval
```
score(t, D) = IDF(t) * tf*(k1+1) / (tf + k1*(1 - b + b*dl/avdl))
IDF(t)      = log((N - df + 0.5) / (df + 0.5))
```

Default: `k1=1.2`, `b=0.75`. `doc_length` and `avdl` are pre-computed at index time.

### 3. WAND Top-K Retrieval

Skips documents that cannot enter the top-K heap using per-term BM25 **upper bounds**
stored as the 5th element of each term's `postings_dict` entry:
```
termID -> (start_pos, num_postings, len_postings_bytes, len_tf_bytes, upper_bound)
```

Upper bounds are computed in one pass after merge (`_compute_upper_bounds`) and persisted to disk.

For each candidate document:
- Sum upper bounds of query terms whose pointer has not passed this doc
- If sum ≤ theta (current K-th best score) → **skip**
- Otherwise → compute exact BM25, update min-heap, update theta

### 4. Evaluation Metrics

Mean scores over 30 queries for both TF-IDF and BM25:

| Metric | Description |
|---|---|
| RBP (p=0.8) | Rank-Biased Precision — rewards early-ranked relevant docs |
| DCG | Discounted Cumulative Gain |
| NDCG | DCG normalized against ideal ranking; range [0, 1] |
| AP | Average Precision — R = total relevant docs in full collection |

Example output:
```
Hasil evaluasi TF-IDF terhadap 30 queries
RBP score  = 0.6451
DCG score  = 5.5818
NDCG score = 0.8182
AP score   = 0.4773
Hasil evaluasi BM25 terhadap 30 queries
RBP score  = 0.6580
DCG score  = 5.6767
NDCG score = 0.8347
AP score   = 0.5029
```

### 5. SPIMI Indexing

`SPIMIIndex` extends `BSBIIndex` with an alternative indexing scheme.
Unlike BSBI which collects all TD pairs first then sorts, SPIMI directly
builds the inverted dictionary token by token as documents are read.

## Author

Ratu Nadya Anjania [2206029752] — [[anjanianadya]](https://github.com/anjanianadya)