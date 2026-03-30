import math
import re
from bsbi import BSBIIndex
from compression import VBEPostings

######## >>>>> sebuah IR metric: RBP p = 0.8

def rbp(ranking, p = 0.8):
  """ menghitung search effectiveness metric score dengan 
      Rank Biased Precision (RBP)

      Parameters
      ----------
      ranking: List[int]
         vektor biner seperti [1, 0, 1, 1, 1, 0]
         gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
         Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                 di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                 di rank-6 tidak relevan
        
      Returns
      -------
      Float
        score RBP
  """
  score = 0.
  for i in range(1, len(ranking) + 1): # added +1 so that the last doc is included in the loop
    pos = i - 1
    score += ranking[pos] * (p ** (i - 1))
  return (1 - p) * score

def dcg(ranking):
    """
    Calculates Discounted Cumulative Gain (DCG).

    Parameters
    ----------
    ranking : List[int]
        Binary relevance vector, ex: [1, 0, 1, 1, 0].

    Returns
    -------
    float
        DCG score.
    """
    score = 0.
    for i in range(1, len(ranking) + 1):
        score += ranking[i - 1] / math.log2(i + 1)
    return score

def ndcg(ranking):
    """
    Calculates Normalized Discounted Cumulative Gain (NDCG).

    Parameters
    ----------
    ranking : List[int]
        Binary relevance vector, ex: [1, 0, 1, 1, 0].

    Returns
    -------
    float
        NDCG score in range [0, 1].
    """
    actual_dcg = dcg(ranking)
    ideal_dcg = dcg(sorted(ranking, reverse=True))

    if ideal_dcg == 0:
        return 0.
    return actual_dcg / ideal_dcg

def ap(ranking, total_relevant):
    """
    Calculates Average Precision (AP).
    Using R from the full qrels rather than sum(ranking) to prevent
    overestimating AP when relevant documents exist beyond rank K.

    Parameters
    ----------
    ranking : List[int]
        Binary relevance vector for the retrieved top-K documents.
    total_relevant : int
        Total number of relevant documents for this query in the
        entire collection, obtained from qrels.

    Returns
    -------
    float
        AP score in range [0, 1].
    """
    if total_relevant == 0:
        return 0.

    score = 0.
    relevant_seen = 0
    for i in range(1, len(ranking) + 1):
        if ranking[i - 1] == 1:
            relevant_seen += 1
            score += relevant_seen / i  # precision at rank i

    return score / total_relevant

######## >>>>> memuat qrels

def load_qrels(qrel_file = "qrels.txt", max_q_id = 30, max_doc_id = 1033):
  """ memuat query relevance judgment (qrels) 
      dalam format dictionary of dictionary
      qrels[query id][document id]

      dimana, misal, qrels["Q3"][12] = 1 artinya Doc 12
      relevan dengan Q3; dan qrels["Q3"][10] = 0 artinya
      Doc 10 tidak relevan dengan Q3.

  """
  qrels = {"Q" + str(q_id) : {doc_id:0 for doc_id in range(1, max_doc_id + 1)} \
                 for q_id in range(1, max_q_id + 1)} # to prevent unambiguity, specified the variables
  with open(qrel_file) as file:
    for line in file:
      parts = line.strip().split()
      qid = parts[0]
      did = int(parts[1])
      qrels[qid][did] = 1
  return qrels

######## >>>>> EVALUASI !

def eval(qrels, query_file = "queries.txt", k = 1000):
  """ 
    loop ke semua 30 query, hitung score di setiap query,
    lalu hitung MEAN SCORE over those 30 queries.
    untuk setiap query, kembalikan top-1000 documents
  """
  BSBI_instance = BSBIIndex(data_dir = 'collection', \
                          postings_encoding = VBEPostings, \
                          output_dir = 'index')

  def run_eval(retrieve_fn, scoring_label):
    """
    Runs the scoring loop for a given retrieval function and prints
    mean scores across all queries.

    Parameters
    ----------
    retrieve_fn : callable
        The retrieval function to evaluate, ex: retrieve_tfidf or retrieve_bm25.
    scoring_label : str
        Display label for this run, ex: "TF-IDF" or "BM25".
    """
    rbp_scores  = []
    dcg_scores  = []
    ndcg_scores = []
    ap_scores   = []

    with open(query_file) as file:
      for qline in file:
        parts = qline.strip().split()
        qid = parts[0]
        query = " ".join(parts[1:])
        total_relevant = sum(qrels[qid].values())

        # HATI-HATI, doc id saat indexing bisa jadi berbeda dengan doc id
        # yang tertera di qrels
        ranking = []
        for (score, doc) in retrieve_fn(query, k = k):
            did = int(re.search(r'\/.*\/.*\/(.*)\.txt', doc).group(1))
            ranking.append(qrels[qid][did])

        rbp_scores.append(rbp(ranking))
        dcg_scores.append(dcg(ranking))
        ndcg_scores.append(ndcg(ranking))
        ap_scores.append(ap(ranking, total_relevant))

    n = len(rbp_scores)
    print(f"Hasil evaluasi {scoring_label} terhadap 30 queries")
    print("RBP score =", sum(rbp_scores) / n)
    print("DCG score =", sum(dcg_scores) / n)
    print("NDCG score =", sum(ndcg_scores) / n)
    print("AP score =", sum(ap_scores) / n)
  run_eval(BSBI_instance.retrieve_tfidf, "TF-IDF")
  run_eval(BSBI_instance.retrieve_bm25,  "BM25")

if __name__ == '__main__':
  qrels = load_qrels()

  assert qrels["Q1"][166] == 1, "qrels salah"
  assert qrels["Q1"][300] == 0, "qrels salah"

  eval(qrels)