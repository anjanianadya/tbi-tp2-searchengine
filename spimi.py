import os
import contextlib

from bsbi import BSBIIndex, STOP_WORDS, STEMMER
from tqdm import tqdm
from index import InvertedIndexReader, InvertedIndexWriter

class SPIMIIndex(BSBIIndex):
    """
    Indexing with SPIMI scheme (Single-Pass In-Memory Indexing).
    Inherits from BSBIIndex, adds index_spimi() and _spimi_invert()
    as alternatives to the BSBI indexing scheme.
    """
    def index_spimi(self):
        os.makedirs(self.output_dir, exist_ok=True)

        for block_dir_relative in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            index_id = 'intermediate_index_' + block_dir_relative
            self.intermediate_indices.append(index_id)

            with InvertedIndexWriter(index_id, self.postings_encoding,
                                    directory=self.output_dir) as index:
                self._spimi_invert(block_dir_relative, index)

        self.save()

        with InvertedIndexWriter(self.index_name, self.postings_encoding,
                                directory=self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(
                    InvertedIndexReader(index_id, self.postings_encoding,
                                        directory=self.output_dir))
                        for index_id in self.intermediate_indices]
                self.merge(indices, merged_index)

        self._compute_upper_bounds(self.index_name)

    def _spimi_invert(self, block_dir_relative, index):
        """
        Core SPIMI inversion for one block.
        Directly creates dictionary and postings list per token
        without needing to collect and sort TD pairs first.

        Parameters
        ----------
        block_dir_relative : str
            Relative path to the block directory.
        index : InvertedIndexWriter
            Inverted index writer for this block.
        """
        dir = "./" + self.data_dir + "/" + block_dir_relative

        term_dict = {}
        term_tf   = {}

        for filename in next(os.walk(dir))[2]:
            docname = dir + "/" + filename
            with open(docname, "r", encoding="utf8", errors="surrogateescape") as f:
                content = f.read().split()
                doc_id = self.doc_id_map[docname]
                for token in content:
                    token = token.lower()
                    if token not in STOP_WORDS:
                        token = STEMMER.stem(token)
                        term_id = self.term_id_map[token]

                        if term_id not in term_dict:
                            term_dict[term_id] = []
                            term_tf[term_id]   = {}
                        if doc_id not in term_tf[term_id]:
                            term_dict[term_id].append(doc_id)
                            term_tf[term_id][doc_id] = 0
                        term_tf[term_id][doc_id] += 1

        # Sort only at the final stage per block (for merging purposes)
        for term_id in sorted(term_dict.keys()):
            sorted_doc_ids = sorted(term_dict[term_id])
            assoc_tf = [term_tf[term_id][doc_id] for doc_id in sorted_doc_ids]
            index.append(term_id, sorted_doc_ids, assoc_tf)

if __name__ == "__main__":
    from compression import VBEPostings

    SPIMI_instance = SPIMIIndex(data_dir='collection',
                                postings_encoding=VBEPostings,
                                output_dir='index_spimi')
    SPIMI_instance.index_spimi()