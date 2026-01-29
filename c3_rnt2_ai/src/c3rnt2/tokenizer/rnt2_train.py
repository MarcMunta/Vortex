from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

from collections import Counter

from .rnt2_model import RNT2Codebook, RNT2Model
from .vortex_tok import VortexMacroCodebook, VortexTokModel


def iter_corpus_files(corpus_dir: Path) -> Iterable[Path]:
    for path in corpus_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".txt", ".md", ".json", ".py"}:
            yield path


def iter_blocks(corpus_dir: Path, block_size: int) -> Iterable[bytes]:
    for path in iter_corpus_files(corpus_dir):
        data = path.read_bytes()
        for i in range(0, len(data), block_size):
            block = data[i : i + block_size]
            if block:
                yield block.ljust(block_size, b"\x00")


def _iter_patch_ids(corpus_dir: Path, block_size: int, codebook: RNT2Codebook) -> Iterable[int]:
    for path in iter_corpus_files(corpus_dir):
        data = path.read_bytes()
        for i in range(0, len(data), block_size):
            block = data[i : i + block_size]
            if not block:
                continue
            padded = block.ljust(block_size, b"\x00")
            code = codebook.find(padded)
            if code is not None:
                yield code


def _iter_patch_sequences(corpus_dir: Path, block_size: int, codebook: RNT2Codebook) -> List[List[int]]:
    sequences: List[List[int]] = []
    for path in iter_corpus_files(corpus_dir):
        data = path.read_bytes()
        seq: List[int] = []
        for i in range(0, len(data), block_size):
            block = data[i : i + block_size]
            if not block:
                continue
            padded = block.ljust(block_size, b"\x00")
            code = codebook.find(padded)
            if code is not None:
                seq.append(code)
        if seq:
            sequences.append(seq)
    return sequences


def build_macro_codebook(
    corpus_dir: Path,
    block_size: int,
    codebook: RNT2Codebook,
    macro_size: int,
    macro_min_len: int,
) -> VortexMacroCodebook:
    if macro_size <= 0:
        return VortexMacroCodebook(sequences=[])
    sequences = _iter_patch_sequences(corpus_dir, block_size, codebook)
    if sum(len(s) for s in sequences) < macro_min_len:
        return VortexMacroCodebook(sequences=[])

    # Build BPE-style merges over patch ids
    symbols = [seq[:] for seq in sequences]
    merges: List[List[int]] = []

    for _ in range(macro_size):
        pair_counts: Counter[tuple[int, int]] = Counter()
        for seq in symbols:
            for i in range(len(seq) - 1):
                pair_counts[(seq[i], seq[i + 1])] += 1
        if not pair_counts:
            break
        (a, b), _ = pair_counts.most_common(1)[0]
        merged = [a, b]
        merges.append(merged)

        # Apply merge to sequences
        new_symbols: List[List[int]] = []
        for seq in symbols:
            i = 0
            new_seq: List[int] = []
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                    # represent merged pair by a negative id offset
                    new_seq.append(-(len(merges)))
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            new_symbols.append(new_seq)
        symbols = new_symbols

    # Expand merges into macro sequences (convert negative ids back)
    def expand(token: int) -> List[int]:
        if token >= 0:
            return [token]
        idx = -token - 1
        seq = merges[idx]
        out: List[int] = []
        for t in seq:
            out.extend(expand(t))
        return out

    sequences: List[List[int]] = []
    for merge in merges:
        expanded: List[int] = []
        for t in merge:
            expanded.extend(expand(t))
        if len(expanded) >= macro_min_len:
            sequences.append(expanded)
    return VortexMacroCodebook(sequences=sequences[:macro_size])


def train(
    codebook_size: int,
    block_size: int,
    corpus_dir: Path,
    output_path: Path,
    vortex_output: Path | None = None,
    macro_size: int = 0,
    macro_min_len: int = 2,
    sub_block_size: int = 0,
    sub_codebook_size: int = 0,
    sub_block_sizes: List[int] | None = None,
    sub_codebook_sizes: List[int] | None = None,
) -> None:
    blocks = list(iter_blocks(corpus_dir, block_size))
    if not blocks:
        codebook = RNT2Codebook.from_builtin(block_size=block_size, size=codebook_size)
    else:
        codebook = RNT2Codebook.from_corpus(blocks=blocks, block_size=block_size, size=codebook_size)
    model = RNT2Model(codebook=codebook)
    model.save(output_path)
    if vortex_output is not None:
        macro = build_macro_codebook(corpus_dir, block_size, codebook, macro_size, macro_min_len)
        sub = None
        sub_codebooks = None
        if sub_block_sizes:
            sub_codebooks = []
            for idx, sb in enumerate(sub_block_sizes):
                if sb and sb < block_size:
                    sub_blocks = list(iter_blocks(corpus_dir, sb))
                    if not sub_blocks:
                        continue
                    size = None
                    if sub_codebook_sizes and idx < len(sub_codebook_sizes):
                        size = int(sub_codebook_sizes[idx])
                    if not size:
                        size = sub_codebook_size or max(128, codebook_size // 4)
                    sub_codebooks.append(RNT2Codebook.from_corpus(blocks=sub_blocks, block_size=sb, size=size))
        elif sub_block_size and sub_block_size < block_size:
            sub_blocks = list(iter_blocks(corpus_dir, sub_block_size))
            if sub_blocks:
                size = sub_codebook_size or max(128, codebook_size // 4)
                sub = RNT2Codebook.from_corpus(blocks=sub_blocks, block_size=sub_block_size, size=size)
        vortex = VortexTokModel(patch_codebook=codebook, macro_codebook=macro, sub_codebook=sub, sub_codebooks=sub_codebooks)
        vortex.save(vortex_output)
    print({
        "blocks": len(blocks),
        "block_size": block_size,
        "codebook_size": codebook_size,
        "output": str(output_path),
        "vortex_output": str(vortex_output) if vortex_output else None,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("data/corpora"))
    parser.add_argument("--output", type=Path, default=Path("data/runs/rnt2_dev.pt"))
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--codebook-size", type=int, default=1024)
    parser.add_argument("--vortex-output", type=Path, default=Path("data/runs/vortex_tok.pt"))
    parser.add_argument("--macro-size", type=int, default=256)
    parser.add_argument("--macro-min-len", type=int, default=2)
    parser.add_argument("--sub-block-size", type=int, default=16)
    parser.add_argument("--sub-codebook-size", type=int, default=256)
    parser.add_argument("--sub-block-sizes", type=str, default=None)
    parser.add_argument("--sub-codebook-sizes", type=str, default=None)
    args = parser.parse_args()
    sub_block_sizes = [int(x) for x in args.sub_block_sizes.split(",")] if args.sub_block_sizes else None
    sub_codebook_sizes = [int(x) for x in args.sub_codebook_sizes.split(",")] if args.sub_codebook_sizes else None
    train(
        args.codebook_size,
        args.block_size,
        args.corpus,
        args.output,
        vortex_output=args.vortex_output,
        macro_size=args.macro_size,
        macro_min_len=args.macro_min_len,
        sub_block_size=args.sub_block_size,
        sub_codebook_size=args.sub_codebook_size,
        sub_block_sizes=sub_block_sizes,
        sub_codebook_sizes=sub_codebook_sizes,
    )


if __name__ == "__main__":
    main()
