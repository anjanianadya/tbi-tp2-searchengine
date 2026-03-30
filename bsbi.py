import os
import pickle
import contextlib
import heapq
import math
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from index import InvertedIndexReader, InvertedIndexWriter
from util import IdMap, sorted_merge_posts_and_tfs
from compression import StandardPostings, VBEPostings
from tqdm import tqdm

nltk.download('stopwords', quiet=True)

STOP_WORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()

class BSBIIndex:
    """
    Attributes
    ----------
    term_id_map(IdMap): Untuk mapping terms ke termIDs
    doc_id_map(IdMap): Untuk mapping relative paths dari dokumen (misal,
                    /collection/0/gamma.txt) to docIDs
    data_dir(str): Path ke data
    output_dir(str): Path ke output index files
    postings_encoding: Lihat di compression.py, kandidatnya adalah StandardPostings,
                    VBEPostings, dsb.
    index_name(str): Nama dari file yang berisi inverted index
    """
    def __init__(self, data_dir, output_dir, postings_encoding, index_name = "main_index"):
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding

        # Untuk menyimpan nama-nama file dari semua intermediate inverted index
        self.intermediate_indices = []

    def save(self):
        """Menyimpan doc_id_map and term_id_map ke output directory via pickle"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'wb') as f:
            pickle.dump(self.term_id_map, f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)

    def load(self):
        """Memuat doc_id_map and term_id_map dari output directory"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'rb') as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)

    def parse_block(self, block_dir_relative):
        """
        Lakukan parsing terhadap text file sehingga menjadi sequence of
        <termID, docID> pairs.

        Gunakan tools available untuk Stemming Bahasa Inggris

        JANGAN LUPA BUANG STOPWORDS!

        Untuk "sentence segmentation" dan "tokenization", bisa menggunakan
        regex atau boleh juga menggunakan tools lain yang berbasis machine
        learning.

        Parameters
        ----------
        block_dir_relative : str
            Relative Path ke directory yang mengandung text files untuk sebuah block.

            CATAT bahwa satu folder di collection dianggap merepresentasikan satu block.
            Konsep block di soal tugas ini berbeda dengan konsep block yang terkait
            dengan operating systems.

        Returns
        -------
        List[Tuple[Int, Int]]
            Returns all the td_pairs extracted from the block
            Mengembalikan semua pasangan <termID, docID> dari sebuah block (dalam hal
            ini sebuah sub-direktori di dalam folder collection)

        Harus menggunakan self.term_id_map dan self.doc_id_map untuk mendapatkan
        termIDs dan docIDs. Dua variable ini harus 'persist' untuk semua pemanggilan
        parse_block(...).
        """
        dir = "./" + self.data_dir + "/" + block_dir_relative
        td_pairs = []
        for filename in next(os.walk(dir))[2]:
            docname = dir + "/" + filename
            with open(docname, "r", encoding = "utf8", errors = "surrogateescape") as f:
                content = f.read().split()
                doc_id = self.doc_id_map[docname]
                for token in content:
                    token = token.lower()
                    if token not in STOP_WORDS:
                        token = STEMMER.stem(token)
                        td_pairs.append((self.term_id_map[token], doc_id))
        return td_pairs

    def invert_write(self, td_pairs, index):
        """
        Melakukan inversion td_pairs (list of <termID, docID> pairs) dan
        menyimpan mereka ke index. Disini diterapkan konsep BSBI dimana 
        hanya di-mantain satu dictionary besar untuk keseluruhan block.
        Namun dalam teknik penyimpanannya digunakan srategi dari SPIMI
        yaitu penggunaan struktur data hashtable (dalam Python bisa
        berupa Dictionary)

        ASUMSI: td_pairs CUKUP di memori

        Di Tugas Pemrograman 1, kita hanya menambahkan term dan
        juga list of sorted Doc IDs. Sekarang di Tugas Pemrograman 2,
        kita juga perlu tambahkan list of TF.

        Parameters
        ----------
        td_pairs: List[Tuple[Int, Int]]
            List of termID-docID pairs
        index: InvertedIndexWriter
            Inverted index pada disk (file) yang terkait dengan suatu "block"
        """
        term_dict = {}
        term_tf = {}
        for term_id, doc_id in td_pairs:
            if term_id not in term_dict:
                term_dict[term_id] = set()
                term_tf[term_id] = {}
            term_dict[term_id].add(doc_id)
            if doc_id not in term_tf[term_id]:
                term_tf[term_id][doc_id] = 0
            term_tf[term_id][doc_id] += 1
        for term_id in sorted(term_dict.keys()):
            sorted_doc_id = sorted(list(term_dict[term_id]))
            assoc_tf = [term_tf[term_id][doc_id] for doc_id in sorted_doc_id]
            index.append(term_id, sorted_doc_id, assoc_tf)

    def merge(self, indices, merged_index):
        """
        Lakukan merging ke semua intermediate inverted indices menjadi
        sebuah single index.

        Ini adalah bagian yang melakukan EXTERNAL MERGE SORT

        Gunakan fungsi orted_merge_posts_and_tfs(..) di modul util

        Parameters
        ----------
        indices: List[InvertedIndexReader]
            A list of intermediate InvertedIndexReader objects, masing-masing
            merepresentasikan sebuah intermediate inveted index yang iterable
            di sebuah block.

        merged_index: InvertedIndexWriter
            Instance InvertedIndexWriter object yang merupakan hasil merging dari
            semua intermediate InvertedIndexWriter objects.
        """
        # kode berikut mengasumsikan minimal ada 1 term
        merged_iter = heapq.merge(*indices, key = lambda x: x[0])
        curr, postings, tf_list = next(merged_iter) # first item
        for t, postings_, tf_list_ in merged_iter: # from the second item
            if t == curr:
                zip_p_tf = sorted_merge_posts_and_tfs(list(zip(postings, tf_list)), \
                                                      list(zip(postings_, tf_list_)))
                postings = [doc_id for (doc_id, _) in zip_p_tf]
                tf_list = [tf for (_, tf) in zip_p_tf]
            else:
                merged_index.append(curr, postings, tf_list)
                curr, postings, tf_list = t, postings_, tf_list_
        merged_index.append(curr, postings, tf_list)

    def _compute_upper_bounds(self, index_name, k1=1.2, b=0.75):
        """
        Computes and stores the BM25 upper bound score for every term in the
        merged index.

        Parameters
        ----------
        index_name : str
            Name of the merged index to update.
        k1 : float
            BM25 term frequency saturation parameter.
        b : float
            BM25 document length normalisation parameter.
        """
        with InvertedIndexReader(index_name, self.postings_encoding,
                                directory=self.output_dir) as merged_index:
            N = len(merged_index.doc_length)
            if N == 0:
                return
            avdl = sum(merged_index.doc_length.values()) / N

            for term in merged_index.terms:
                pos, df, len_post, len_tf, _ = merged_index.postings_dict[term]

                idf = math.log((N - df + 0.5) / (df + 0.5))

                postings, tf_list = merged_index.get_postings_list(term)
                max_term_weight = 0.0
                for doc_id, tf in zip(postings, tf_list):
                    dl = merged_index.doc_length[doc_id]
                    numerator   = tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * (dl / avdl))
                    weight = idf * (numerator / denominator)
                    if weight > max_term_weight:
                        max_term_weight = weight

                merged_index.postings_dict[term] = (pos, df, len_post,
                                                    len_tf, max_term_weight)

    def preprocess_query(self, query):
        """
        A helper method that applies the same preprocessing pipeline used in parse_block to a
        query string: lowercasing, stopword removal, and stemming.

        Parameters
        ----------
        query : str
            Raw query string.

        Returns
        -------
        List[str]
            List of preprocessed query tokens.
        """
        tokens = []
        for token in query.split():
            token = token.lower()
            if token not in STOP_WORDS:
                token = STEMMER.stem(token)
                tokens.append(token)
        return tokens

    def retrieve_tfidf(self, query, k = 10):
        """
        Melakukan Ranked Retrieval dengan skema TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        w(t, D) = (1 + log tf(t, D))       jika tf(t, D) > 0
                = 0                        jika sebaliknya

        w(t, Q) = IDF = log (N / df(t))

        Score = untuk setiap term di query, akumulasikan w(t, Q) * w(t, D).
                (tidak perlu dinormalisasi dengan panjang dokumen)

        catatan: 
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_li
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        terms = []
        for word in self.preprocess_query(query):
            if word in self.term_id_map: # only consider terms that exist in the collection
                terms.append(self.term_id_map[word])
        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:

            scores = {}
            for term in terms:
                if term in merged_index.postings_dict:
                    df = merged_index.postings_dict[term][1]
                    N = len(merged_index.doc_length)
                    postings, tf_list = merged_index.get_postings_list(term)
                    for i in range(len(postings)):
                        doc_id, tf = postings[i], tf_list[i]
                        if doc_id not in scores:
                            scores[doc_id] = 0
                        if tf > 0:
                            scores[doc_id] += math.log(N / df) * (1 + math.log(tf))

            # Top-K
            docs = [(score, self.doc_id_map[doc_id]) for (doc_id, score) in scores.items()]
            return sorted(docs, key = lambda x: x[0], reverse = True)[:k]

    def retrieve_bm25(self, query, k=10, k1=1.2, b=0.75):
        """
        Performs Ranked Retrieval using the BM25 scoring scheme (Term-at-a-Time).
        Returns the top-K documents sorted descending based on BM25 score.

        Parameters
        ----------
        query : str
            Space-separated query string.
        k : int
            Num of top results to return. Default is 10.
        k1 : float
            Controls how quickly TF weight saturates.
        b : float
            Controls document length normalisation strength.

        Returns
        -------
        List[Tuple[float, str]]
            List of (score, document_name) tuples, sorted by score descending.
        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            N = len(merged_index.doc_length)

            if N == 0:
                return []

            avdl = sum(merged_index.doc_length.values()) / N

            scores = {}
            for word in self.preprocess_query(query):
                if word not in self.term_id_map:
                    continue
                term_id = self.term_id_map[word]

                if term_id not in merged_index.postings_dict:
                    continue

                df = merged_index.postings_dict[term_id][1]

                idf = math.log((N - df + 0.5) / (df + 0.5))

                postings, tf_list = merged_index.get_postings_list(term_id)
                for doc_id, tf in zip(postings, tf_list):
                    dl = merged_index.doc_length[doc_id]
                    if doc_id not in scores:
                        scores[doc_id] = 0.0

                    # BM25 term weight
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * (dl / avdl))
                    scores[doc_id] += idf * (numerator / denominator)

            docs = [(score, self.doc_id_map[doc_id]) for doc_id, score in scores.items()]
            return sorted(docs, key=lambda x: x[0], reverse=True)[:k]

    def retrieve_bm25_wand(self, query, k=10, k1=1.2, b=0.75):
        """
        Performs Top-K retrieval using the WAND (Weak AND) algorithm with
        BM25 scoring.

        Parameters
        ----------
        query : str
            Space-separated query string.
        k : int
            Number of top results to return.
        k1 : float
            BM25 term frequency saturation parameter.
        b : float
            BM25 document length normalisation parameter.

        Returns
        -------
        List[Tuple[float, str]]
            List of (score, document_name) tuples, sorted by score descending.
        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        with InvertedIndexReader(self.index_name, self.postings_encoding,
                                directory=self.output_dir) as merged_index:
            N = len(merged_index.doc_length)
            if N == 0:
                return []

            avdl = sum(merged_index.doc_length.values()) / N

            # Build per-term data structures
            term_data = []
            for word in self.preprocess_query(query):
                if word not in self.term_id_map:
                    continue
                term_id = self.term_id_map[word]
                if term_id not in merged_index.postings_dict:
                    continue

                _, df, _, _, upper_bound = merged_index.postings_dict[term_id]
                idf = math.log((N - df + 0.5) / (df + 0.5))
                postings, tf_list = merged_index.get_postings_list(term_id)
                term_data.append({
                    'upper_bound': upper_bound,
                    'idf': idf,
                    'postings': postings,
                    'tf_list': tf_list,
                    'ptr': 0
                })

            if not term_data:
                return []

            heap = []
            theta = 0.0

            # Collect all unique candidate doc IDs across all postings lists and sort them
            all_doc_ids = sorted(set(
                doc_id
                for td in term_data
                for doc_id in td['postings']
            ))

            for doc_id in all_doc_ids:
                # WAND upper bound check
                ub_sum = sum(
                    td['upper_bound']
                    for td in term_data
                    if td['ptr'] < len(td['postings'])
                    and td['postings'][td['ptr']] == doc_id
                )

                # If the upper bound sum cannot beat the current threshold,
                # skip this document entirely, no need to compute exact score
                if ub_sum <= theta:
                    # Advance pointers past this doc_id
                    for td in term_data:
                        while td['ptr'] < len(td['postings']) \
                            and td['postings'][td['ptr']] <= doc_id:
                            td['ptr'] += 1
                    continue

                dl = merged_index.doc_length[doc_id]
                exact_score = 0.0
                for td in term_data:
                    # Find doc_id in the term's postings list
                    ptr = td['ptr']
                    while ptr < len(td['postings']) \
                        and td['postings'][ptr] < doc_id:
                        ptr += 1
                    if ptr < len(td['postings']) \
                    and td['postings'][ptr] == doc_id:
                        tf = td['tf_list'][ptr]
                        numerator   = tf * (k1 + 1)
                        denominator = tf + k1 * (1 - b + b * (dl / avdl))
                        exact_score += td['idf'] * (numerator / denominator)

                # Update top-K heap
                if len(heap) < k:
                    heapq.heappush(heap, (exact_score, doc_id))
                elif exact_score > heap[0][0]:
                    heapq.heapreplace(heap, (exact_score, doc_id))

                # Update threshold = smallest score in current top-K
                if len(heap) == k:
                    theta = heap[0][0]

                # Advance pointers past this doc_id
                for td in term_data:
                    while td['ptr'] < len(td['postings']) \
                        and td['postings'][td['ptr']] <= doc_id:
                        td['ptr'] += 1

            docs = [(score, self.doc_id_map[doc_id]) for score, doc_id in heap]
            return sorted(docs, key=lambda x: x[0], reverse=True)
        
    def index(self):
        """
        Base indexing code
        BAGIAN UTAMA untuk melakukan Indexing dengan skema BSBI (blocked-sort
        based indexing)

        Method ini scan terhadap semua data di collection, memanggil parse_block
        untuk parsing dokumen dan memanggil invert_write yang melakukan inversion
        di setiap block dan menyimpannya ke index yang baru.
        """
        # loop untuk setiap sub-directory di dalam folder collection (setiap block)
        for block_dir_relative in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            td_pairs = self.parse_block(block_dir_relative)
            index_id = 'intermediate_index_'+block_dir_relative
            self.intermediate_indices.append(index_id)
            with InvertedIndexWriter(index_id, self.postings_encoding, directory = self.output_dir) as index:
                self.invert_write(td_pairs, index)
                td_pairs = None
    
        self.save()

        with InvertedIndexWriter(self.index_name, self.postings_encoding, directory = self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(InvertedIndexReader(index_id, self.postings_encoding, directory=self.output_dir))
                               for index_id in self.intermediate_indices]
                self.merge(indices, merged_index)

        # Compute and store upper bounds after merge
        self._compute_upper_bounds(self.index_name)


if __name__ == "__main__":

    BSBI_instance = BSBIIndex(data_dir = 'collection', \
                              postings_encoding = VBEPostings, \
                              output_dir = 'index')
    BSBI_instance.index() # memulai indexing!
