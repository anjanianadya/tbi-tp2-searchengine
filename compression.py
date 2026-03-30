import array

class StandardPostings:
    """ 
    Class dengan static methods, untuk mengubah representasi postings list
    yang awalnya adalah List of integer, berubah menjadi sequence of bytes.
    Kita menggunakan Library array di Python.

    ASUMSI: postings_list untuk sebuah term MUAT di memori!

    Silakan pelajari:
        https://docs.python.org/3/library/array.html
    """

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list menjadi stream of bytes

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (postings)

        Returns
        -------
        bytes
            bytearray yang merepresentasikan urutan integer di postings_list
        """
        # Untuk yang standard, gunakan L untuk unsigned long, karena docID
        # tidak akan negatif. Dan kita asumsikan docID yang paling besar
        # cukup ditampung di representasi 4 byte unsigned.
        return array.array('L', postings_list).tobytes()

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decodes postings_list dari sebuah stream of bytes

        Parameters
        ----------
        encoded_postings_list: bytes
            bytearray merepresentasikan encoded postings list sebagai keluaran
            dari static method encode di atas.

        Returns
        -------
        List[int]
            list of docIDs yang merupakan hasil decoding dari encoded_postings_list
        """
        decoded_postings_list = array.array('L')
        decoded_postings_list.frombytes(encoded_postings_list)
        return decoded_postings_list.tolist()

    @staticmethod
    def encode_tf(tf_list):
        """
        Encode list of term frequencies menjadi stream of bytes

        Parameters
        ----------
        tf_list: List[int]
            List of term frequencies

        Returns
        -------
        bytes
            bytearray yang merepresentasikan nilai raw TF kemunculan term di setiap
            dokumen pada list of postings
        """
        return StandardPostings.encode(tf_list)

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decodes list of term frequencies dari sebuah stream of bytes

        Parameters
        ----------
        encoded_tf_list: bytes
            bytearray merepresentasikan encoded term frequencies list sebagai keluaran
            dari static method encode_tf di atas.

        Returns
        -------
        List[int]
            List of term frequencies yang merupakan hasil decoding dari encoded_tf_list
        """
        return StandardPostings.decode(encoded_tf_list)

class VBEPostings:
    """ 
    Berbeda dengan StandardPostings, dimana untuk suatu postings list,
    yang disimpan di disk adalah sequence of integers asli dari postings
    list tersebut apa adanya.

    Pada VBEPostings, kali ini, yang disimpan adalah gap-nya, kecuali
    posting yang pertama. Barulah setelah itu di-encode dengan Variable-Byte
    Enconding algorithm ke bytestream.

    Contoh:
    postings list [34, 67, 89, 454] akan diubah dulu menjadi gap-based,
    yaitu [34, 33, 22, 365]. Barulah setelah itu di-encode dengan algoritma
    compression Variable-Byte Encoding, dan kemudian diubah ke bytesream.

    ASUMSI: postings_list untuk sebuah term MUAT di memori!

    """

    @staticmethod
    def vb_encode_number(number):
        """
        Encodes a number using Variable-Byte Encoding
        Lihat buku teks kita!
        """
        bytes = []
        while True:
            bytes.insert(0, number % 128) # prepend ke depan
            if number < 128:
                break
            number = number // 128
        bytes[-1] += 128 # bit awal pada byte terakhir diganti 1
        return array.array('B', bytes).tobytes()

    @staticmethod
    def vb_encode(list_of_numbers):
        """ 
        Melakukan encoding (tentunya dengan compression) terhadap
        list of numbers, dengan Variable-Byte Encoding
        """
        bytes = []
        for number in list_of_numbers:
            bytes.append(VBEPostings.vb_encode_number(number))
        return b"".join(bytes)

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list menjadi stream of bytes (dengan Variable-Byte
        Encoding). JANGAN LUPA diubah dulu ke gap-based list, sebelum
        di-encode dan diubah ke bytearray.

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (postings)

        Returns
        -------
        bytes
            bytearray yang merepresentasikan urutan integer di postings_list
        """
        gap_postings_list = [postings_list[0]]
        for i in range(1, len(postings_list)):
            gap_postings_list.append(postings_list[i] - postings_list[i-1])
        return VBEPostings.vb_encode(gap_postings_list)

    @staticmethod
    def encode_tf(tf_list):
        """
        Encode list of term frequencies menjadi stream of bytes

        Parameters
        ----------
        tf_list: List[int]
            List of term frequencies

        Returns
        -------
        bytes
            bytearray yang merepresentasikan nilai raw TF kemunculan term di setiap
            dokumen pada list of postings
        """
        return VBEPostings.vb_encode(tf_list)

    @staticmethod
    def vb_decode(encoded_bytestream):
        """
        Decoding sebuah bytestream yang sebelumnya di-encode dengan
        variable-byte encoding.
        """
        n = 0
        numbers = []
        decoded_bytestream = array.array('B')
        decoded_bytestream.frombytes(encoded_bytestream)
        bytestream = decoded_bytestream.tolist()
        for byte in bytestream:
            if byte < 128:
                n = 128 * n + byte
            else:
                n = 128 * n + (byte - 128)
                numbers.append(n)
                n = 0
        return numbers

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decodes postings_list dari sebuah stream of bytes. JANGAN LUPA
        bytestream yang di-decode dari encoded_postings_list masih berupa
        gap-based list.

        Parameters
        ----------
        encoded_postings_list: bytes
            bytearray merepresentasikan encoded postings list sebagai keluaran
            dari static method encode di atas.

        Returns
        -------
        List[int]
            list of docIDs yang merupakan hasil decoding dari encoded_postings_list
        """
        decoded_postings_list = VBEPostings.vb_decode(encoded_postings_list)
        total = decoded_postings_list[0]
        ori_postings_list = [total]
        for i in range(1, len(decoded_postings_list)):
            total += decoded_postings_list[i]
            ori_postings_list.append(total)
        return ori_postings_list

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decodes list of term frequencies dari sebuah stream of bytes

        Parameters
        ----------
        encoded_tf_list: bytes
            bytearray merepresentasikan encoded term frequencies list sebagai keluaran
            dari static method encode_tf di atas.

        Returns
        -------
        List[int]
            List of term frequencies yang merupakan hasil decoding dari encoded_tf_list
        """
        return VBEPostings.vb_decode(encoded_tf_list)
    
class EliasGammaPostings:
    """
    Encodes and decodes postings lists and term frequency lists using
    Elias-Gamma compression at the bit level.

    Elias-Gamma represents each positive integer x as:
    - N leading zero bits (N = floor(log2(x)))
    - followed by the full binary representation of x 
    (with the leading '1' acts as the separator)

    Ex: 6 = 110 in binary, N = 2 → encoded as '00110'

    Gap encoding is applied to postings lists before compression,
    so only the first docID is stored as-is. The rest are stored
    as differences from the previous docID.

    ASSUMPTION: the postings list for any single term fits in memory.
    """

    @staticmethod
    def elias_gamma_encode_number(n):
        """
        Encodes a single positive integer using Elias-Gamma encoding.

        Parameters
        ----------
        n : int
            A positive integer to encode. Must be >= 1.

        Returns
        -------
        str
            A binary string representing the Elias-Gamma encoding of n.
        """
        if n <= 0:
            raise ValueError(
                f"Elias-Gamma encoding is only for positive integers, yet got {n} instead."
            )
        binary = bin(n)[2:]
        N = len(binary) - 1
        unary = "0" * N

        return unary + binary

    @staticmethod
    def encode(postings_list):
        """
        Encodes a postings list into a compressed byte string.

        Parameters
        ----------
        postings_list : list of int
            A sorted list of document IDs (docIDs). Must be non-empty,
            and each consecutive pair must be strictly increasing so
            that all gaps are positive integers.

        Returns
        -------
        bytes
            The Elias-Gamma compressed byte representation of the postings list.
        """
        # Convert to gap-based
        gaps = [postings_list[0]]
        for i in range(1, len(postings_list)):
            gaps.append(postings_list[i] - postings_list[i - 1])

        # Encode each gap and join into a single bit string
        bit_string = "".join(EliasGammaPostings.elias_gamma_encode_number(g) for g in gaps)

        # Pad with zeros so the total length is a multiple of 8
        padding_len = (8 - len(bit_string) % 8) % 8
        bit_string += "0" * padding_len

        # Convert each 8-bit chunk into a byte
        byte_list = [int(bit_string[i:i+8], 2) for i in range(0, len(bit_string), 8)]
        return bytes(byte_list)

    @staticmethod
    def decode(encoded_bytes):
        """
        Decodes a compressed byte string back into the original postings list.

        Parameters
        ----------
        encoded_bytes : bytes
            A byte string produced by EliasGammaPostings.encode().

        Returns
        -------
        list of int
            The original sorted list of docIDs.
            Returns an empty list if encoded_bytes yields no valid values.
        """
        # Expand each byte into its 8-bit binary string
        bit_string = "".join(bin(b)[2:].zfill(8) for b in encoded_bytes)

        numbers = []
        i = 0
        while i < len(bit_string):
            # Count leading zeros to find N
            start_i = i
            while i < len(bit_string) and bit_string[i] == '0':
                i += 1

            # Reached the end while counting zeros, the rest is padding
            if i >= len(bit_string):
                break

            N = i - start_i

            # Read N+1 bits starting from the '1' separator
            val_str = bit_string[i : i + N + 1]

            # Fewer bits than expected means the stream is malformed or padded
            if len(val_str) < N + 1:
                break

            numbers.append(int(val_str, 2))
            i += N + 1

        if not numbers:
            return []

        # Reconstruct original docIDs from gap values via cumulative sum
        postings = [numbers[0]]
        for j in range(1, len(numbers)):
            postings.append(postings[j - 1] + numbers[j])
        return postings

    @staticmethod
    def encode_tf(tf_list):
        """
        Encodes a list of term frequencies into a compressed byte string.

        Note: Unlike postings encoding, term frequencies are encoded directly
        without gap conversion, since they are not required to be sorted
        or strictly increasing.

        Parameters
        ----------
        tf_list : list of int
            A list of raw term frequency values. Each value must be >= 1,
            since a term must appear at least once to have a posting.

        Returns
        -------
        bytes
            The Elias-Gamma compressed byte representation of the term frequency list.
        """
        bit_string = "".join(EliasGammaPostings.elias_gamma_encode_number(tf) for tf in tf_list)

        # Pad and pack into bytes, same as encode()
        padding_len = (8 - len(bit_string) % 8) % 8
        bit_string += "0" * padding_len
        return bytes(int(bit_string[i:i+8], 2) for i in range(0, len(bit_string), 8))

    @staticmethod
    def decode_tf(encoded_bytes):
        """
        Decodes a compressed byte string back into the original term frequency list.

        Mirrors the logic of decode(), but without the gap-to-docID reconstruction step.

        Parameters
        ----------
        encoded_bytes : bytes
            A byte string produced by EliasGammaPostings.encode_tf().

        Returns
        -------
        list of int
            The original list of term frequency values.
            Returns an empty list if encoded_bytes yields no valid values.
        """
        # Expand each byte into its 8-bit binary string
        bit_string = "".join(bin(b)[2:].zfill(8) for b in encoded_bytes)

        tfs = []
        i = 0
        while i < len(bit_string):
            # Count leading zeros to find N
            prefix_zeros = 0
            while i < len(bit_string) and bit_string[i] == '0':
                prefix_zeros += 1
                i += 1

            # Reached the end, the rest is only padding zeros
            if i >= len(bit_string):
                break

            # Read N+1 bits to recover the encoded value
            num_bits = prefix_zeros + 1
            tfs.append(int(bit_string[i : i + num_bits], 2))
            i += num_bits

        return tfs

if __name__ == '__main__':
    
    postings_list = [34, 67, 89, 454, 2345738]
    tf_list = [12, 10, 3, 4, 1]
    for Postings in [StandardPostings, VBEPostings, EliasGammaPostings]:
        print(Postings.__name__)
        encoded_postings_list = Postings.encode(postings_list)
        encoded_tf_list = Postings.encode_tf(tf_list)
        print("byte hasil encode postings: ", encoded_postings_list)
        print("ukuran encoded postings   : ", len(encoded_postings_list), "bytes")
        print("byte hasil encode TF list : ", encoded_tf_list)
        print("ukuran encoded TF list    : ", len(encoded_tf_list), "bytes")
        
        decoded_posting_list = Postings.decode(encoded_postings_list)
        decoded_tf_list = Postings.decode_tf(encoded_tf_list)
        print("hasil decoding (postings): ", decoded_posting_list)
        print("hasil decoding (TF list) : ", decoded_tf_list)
        assert decoded_posting_list == postings_list, "hasil decoding tidak sama dengan postings original"
        assert decoded_tf_list == tf_list, "hasil decoding tidak sama dengan postings original"
        print()
