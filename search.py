import time

from bsbi import BSBIIndex
from compression import VBEPostings

# sebelumnya sudah dilakukan indexing
# BSBIIndex hanya sebagai abstraksi untuk index tersebut
BSBI_instance = BSBIIndex(data_dir = 'collection', \
                          postings_encoding = VBEPostings, \
                          output_dir = 'index')

queries = ["alkylated with radioactive iodoacetate", \
           "psychodrama for disturbed children", \
           "lipid metabolism in toxemia and normal pregnancy"]
           
for query in queries:
    print("Query  : ", query)
    print("Results:")
    for (score, doc) in BSBI_instance.retrieve_tfidf(query, k = 10):
        print(f"{doc:30} {score:>.3f}")
    print()

# Sanity check
print("+++ WAND Sanity Check +++")
for query in queries:
    result_bm25 = BSBI_instance.retrieve_bm25(query, k=10)
    result_bm25_wand = BSBI_instance.retrieve_bm25_wand(query, k=10)
    assert result_bm25 == result_bm25_wand, f"WAND mismatch for query: '{query}'"
    print(f"OK: '{query}'")
print("All WAND tests passed.")

# Timing comparison: BM25 vs WAND
print("\n+++ Timing Comparison: BM25 vs WAND +++")
for query in queries:
    start = time.time()
    BSBI_instance.retrieve_bm25(query, k=10)
    bm25_time = time.time() - start

    start = time.time()
    BSBI_instance.retrieve_bm25_wand(query, k=10)
    wand_time = time.time() - start

    print(f"Query: '{query}'")
    print(f"BM25 : {bm25_time:.4f}s")
    print(f"WAND : {wand_time:.4f}s")
    print(f"Speedup: {bm25_time / wand_time:.2f}x")
    print()