import heapq
from typing import Dict, Set, List, Optional, Tuple
from collections import defaultdict
import regex as re

from tqdm import tqdm


class Merge:
    idx0: int
    idx1: int
    __slots__ = ["idx0", "idx1"]

    def __init__(self, idx0, idx1):
        self.idx0 = idx0
        self.idx1 = idx1

    def __hash__(self):
        return self.idx0 * 1000000007 + self.idx1

    def __eq__(self, other: "Merge"):
        return self.idx0 == other.idx0 and self.idx1 == other.idx1


class HeapItem:
    merge: Merge
    count: int
    vocab: Dict[int, str]
    __slots__ = ["merge", "count", "vocab"]

    def __init__(self, merge, count, vocab):
        self.merge = merge
        self.count = count
        self.vocab = vocab

    def __lt__(self, other: "HeapItem"):
        return (
            self.count > other.count
            or self.count == other.count
            and self.vocab[self.merge.idx0] > self.vocab[other.merge.idx0]
            or self.count == other.count
            and self.vocab[self.merge.idx0] == self.vocab[other.merge.idx0]
            and self.vocab[self.merge.idx1] > self.vocab[other.merge.idx1]
        )

    def __str__(self):
        return f"Merge {self.vocab[self.merge.idx0]}, {self.vocab[self.merge.idx1]}, {self.count}"


class Node:
    l: "Node"
    r: "Node"
    idx: int
    __slots__ = ["l", "r", "idx"]


class BPETokenizer:
    vocab: List[bytes]
    merges: List[Merge]

    def __init__(self, special_tokens=None):
        self.special_tokens: Optional[List[str]] = special_tokens

    def encode(self, s: str) -> List[int]: ...

    @staticmethod
    def from_params(
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: Optional[List[str]] = None,
    ) -> "BPETokenizer":
        tokenizer = BPETokenizer()
        rev_vocab = {v: k for k, v in vocab.items()}
        vocab_size = max(vocab.keys()) + 1
        tokenizer.vocab = [vocab.get(i) for i in range(vocab_size)]
        tokenizer.merges = [Merge(rev_vocab[b1], rev_vocab[b2]) for b1, b2 in merges]
        tokenizer.special_tokens = special_tokens

    def export_params(self):
        vocab = {i: self.vocab[i] for i in range(len(self.vocab))}
        merges = [(self.vocab[m.idx0], self.vocab[m.idx1]) for m in self.merges]
        return vocab, merges

    def train(self, text: str, vocab_size: int):
        GPT2_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        if self.special_tokens is not None:
            special_token_pattern = (
                f"({'|'.join(re.escape(s) for s in self.special_tokens)})"
            )
            text_array = re.split(special_token_pattern, text)
            initial_vocab = [s.encode("utf-8") for s in self.special_tokens]
        else:
            text_array = [text]
            initial_vocab = []

        for i in range(256):
            initial_vocab.append(bytes([i]))
        rev_initial_vocab = {v: k for k, v in enumerate(initial_vocab)}

        groups = []
        for text in text_array:
            if text in self.special_tokens or not text:
                continue
            indices_array: List[List[int]] = [
                [rev_initial_vocab[bytes([b])] for b in s.encode("utf-8")]
                for s in re.findall(GPT2_PATTERN, text)
            ]
            groups.extend(indices_array)

        self._merge(groups, vocab_size - len(initial_vocab), initial_vocab)

    def _merge(self, groups, num_merges, initial_vocab):
        num_nodes = sum([len(g) for g in groups])
        nodes = [Node() for _ in range(num_nodes)]
        freq_table: Dict[Merge, int] = defaultdict(int)
        vocab = [None for _ in range(len(initial_vocab) + num_merges)]
        merges = [None for _ in range(num_merges)]
        for i in range(len(initial_vocab)):
            vocab[i] = initial_vocab[i]
        merge_to_nodes: Dict[Merge, Set[Node]] = defaultdict(set)
        offset = 0
        for g in groups:
            for i, idx in enumerate(g):
                nodes[offset + i].idx = idx
                nodes[offset + i].l = None if i == 0 else nodes[offset + i - 1]
                nodes[offset + i].r = None if i + 1 >= len(g) else nodes[offset + i + 1]
                if i + 1 < len(g):
                    merge = Merge(idx, g[i + 1])
                    freq_table[merge] += 1
                    merge_to_nodes[merge].add(nodes[offset + i])
            offset += len(g)
        heap = [HeapItem(k, v, vocab) for k, v in freq_table.items()]
        heapq.heapify(heap)
        for merge_id in tqdm(range(num_merges)):
            while True:
                top = heapq.heappop(heap)
                if freq_table.get(top.merge) is None:
                    continue
                if freq_table[top.merge] != top.count:
                    heapq.heappush(
                        heap, HeapItem(top.merge, freq_table[top.merge], vocab)
                    )
                    continue
                break
            merges[merge_id] = top.merge
            new_idx = merge_id + len(initial_vocab)
            vocab[new_idx] = vocab[top.merge.idx0] + vocab[top.merge.idx1]
            for node in list(merge_to_nodes[top.merge]):
                if not (
                    node.r is not None
                    and node.idx == top.merge.idx0
                    and node.r.idx == top.merge.idx1
                ):
                    # prevent phantom node
                    continue
                if node.l is not None:
                    merge = Merge(node.l.idx, node.idx)
                    merge_to_nodes[merge].discard(node.l)
                    freq_table[merge] -= 1
                    merge = Merge(node.l.idx, new_idx)
                    merge_to_nodes[merge].add(node.l)
                    freq_table[merge] += 1
                    heapq.heappush(heap, HeapItem(merge, freq_table[merge], vocab))
                if node.r.r is not None:
                    merge = Merge(node.r.idx, node.r.r.idx)
                    merge_to_nodes[merge].discard(node.r)
                    freq_table[merge] -= 1
                    merge = Merge(new_idx, node.r.r.idx)
                    merge_to_nodes[merge].add(node)
                    freq_table[merge] += 1
                    heapq.heappush(heap, HeapItem(merge, freq_table[merge], vocab))
                r = node.r
                node.r = r.r
                if r.r is not None:
                    r.r.l = node
                node.idx = new_idx
            del freq_table[top.merge]
            del merge_to_nodes[top.merge]
        self.vocab = vocab
        self.merges = merges
