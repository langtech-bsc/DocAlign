#!/usr/bin/env python3
"""
Debug script to trace where sentence duplication occurs.
Add this to your pipeline after sentence alignment.
"""

def debug_sentence_alignments(all_sentence_alignments):
    """Check for duplicates in the sentence alignment output."""
    print("\n" + "="*60)
    print("DEBUG: Checking sentence alignments for duplicates")
    print("="*60)
    
    # Track which sentence indices appear where
    src_usage = {}  # src_idx -> list of (para_A, para_B, entry_idx)
    tgt_usage = {}  # tgt_idx -> list of (para_A, para_B, entry_idx)
    
    for entry_idx, entry in enumerate(all_sentence_alignments):
        para_A = entry["paragraph_A"]
        para_B = entry["paragraph_B"]
        sent_pairs = entry["sentence_pairs"]
        
        for sent_pair in sent_pairs:
            src_idx = (sent_pair["src_idx_start"], sent_pair["src_idx_end"])
            tgt_idx = (sent_pair["tgt_idx_start"], sent_pair["tgt_idx_end"])
            src_text = sent_pair["src_text"]
            tgt_text = sent_pair["tgt_text"]
            
            # Track usage
            if src_idx not in src_usage:
                src_usage[src_idx] = []
            src_usage[src_idx].append((para_A, para_B, entry_idx, src_text))
            
            if tgt_idx not in tgt_usage:
                tgt_usage[tgt_idx] = []
            tgt_usage[tgt_idx].append((para_A, para_B, entry_idx, tgt_text))
    
    # Find duplicates
    src_duplicates = {idx: locs for idx, locs in src_usage.items() if len(locs) > 1}
    tgt_duplicates = {idx: locs for idx, locs in tgt_usage.items() if len(locs) > 1}
    
    if src_duplicates:
        print("\n⚠ SOURCE DUPLICATES FOUND:")
        for src_idx, locs in src_duplicates.items():
            print(f"\n  src_idx {src_idx} appears {len(locs)} times:")
            for para_A, para_B, entry_idx, text in locs:
                print(f"    - Entry {entry_idx}: para_A={para_A}, para_B={para_B}")
                print(f"      Text: {text[:80]}...")
    else:
        print("\n✓ No source sentence duplicates")
    
    if tgt_duplicates:
        print("\n⚠ TARGET DUPLICATES FOUND:")
        for tgt_idx, locs in tgt_duplicates.items():
            print(f"\n  tgt_idx {tgt_idx} appears {len(locs)} times:")
            for para_A, para_B, entry_idx, text in locs:
                print(f"    - Entry {entry_idx}: para_A={para_A}, para_B={para_B}")
                print(f"      Text: {text[:80]}...")
    else:
        print("\n✓ No target sentence duplicates")
    
    return len(src_duplicates) > 0 or len(tgt_duplicates) > 0


def debug_sentence_alignment_index(sent_index):
    """Check the sentence alignment index for duplicate keys/values."""
    print("\n" + "="*60)
    print("DEBUG: Checking sentence alignment index")
    print("="*60)
    
    print(f"\nTotal keys in index: {len(sent_index)}")
    
    # Track which (src_idx, tgt_idx) pairs appear
    pair_usage = {}
    for (para_A, para_B), sent_pairs in sent_index.items():
        for sent_pair in sent_pairs:
            src_idx = sent_pair["src_idx"]
            tgt_idx = sent_pair["tgt_idx"]
            pair_key = (src_idx, tgt_idx)
            
            if pair_key not in pair_usage:
                pair_usage[pair_key] = []
            pair_usage[pair_key].append((para_A, para_B))
    
    duplicates = {pair: locs for pair, locs in pair_usage.items() if len(locs) > 1}
    
    if duplicates:
        print(f"\n⚠ DUPLICATE PAIRS FOUND: {len(duplicates)}")
        for (src_idx, tgt_idx), locs in list(duplicates.items())[:5]:
            print(f"  (src={src_idx}, tgt={tgt_idx}) appears in {len(locs)} paragraph pairs:")
            for para_A, para_B in locs:
                print(f"    para_A={para_A}, para_B={para_B}")
    else:
        print("\n✓ No duplicate (src, tgt) pairs across different paragraph alignments")


def debug_paragraph_blocks(paragraph_blocks):
    """Check for duplicate sentences in paragraph blocks."""
    print("\n" + "="*60)
    print("DEBUG: Checking paragraph blocks for duplicates")
    print("="*60)
    
    for block_idx, block in enumerate(paragraph_blocks):
        src_text = block["src_block"]
        tgt_text = block["tgt_block"]
        
        # Split into sentences and check for repeats
        src_sents = src_text.split(". ")
        tgt_sents = tgt_text.split(". ")
        
        src_seen = {}
        for sent_idx, sent in enumerate(src_sents):
            if sent in src_seen:
                print(f"\n⚠ Block {block_idx} (para {para_idx}): SOURCE SENTENCE REPEATED")
                print(f"  First at position {src_seen[sent]}, again at {sent_idx}")
                print(f"  Text: {sent[:100]}...")
            else:
                src_seen[sent] = sent_idx
        
        tgt_seen = {}
        for sent_idx, sent in enumerate(tgt_sents):
            if sent in tgt_seen:
                print(f"\n⚠ Block {block_idx} (para {para_idx}): TARGET SENTENCE REPEATED")
                print(f"  First at position {tgt_seen[sent]}, again at {sent_idx}")
                print(f"  Text: {sent[:100]}...")
            else:
                tgt_seen[sent] = sent_idx


def debug_chunks(chunks):
    """Check for duplicate sentences in final chunks."""
    print("\n" + "="*60)
    print("DEBUG: Checking final chunks for duplicates")
    print("="*60)
    
    for chunk_idx, chunk in enumerate(chunks):
        src_text = chunk["src_text"]
        tgt_text = chunk["tgt_text"]
        
        # Simple check: count occurrences of sentences
        src_sents = src_text.split("\n\n")  # chunks use \n\n as separator
        tgt_sents = tgt_text.split("\n\n")
        
        print(f"\nChunk {chunk_idx}:")
        print(f"  Paragraph pairs: {chunk['paragraph_pairs']}")
        print(f"  Source blocks: {len(src_sents)}, Target blocks: {len(tgt_sents)}")
        
        # Check for repeated blocks within a chunk
        seen_src = {}
        for block_idx, block in enumerate(src_sents):
            if block in seen_src and block.strip():  # ignore empty
                print(f"  ⚠ SOURCE BLOCK REPEATED: first at {seen_src[block]}, again at {block_idx}")
                print(f"    Text: {block[:100]}...")
            else:
                seen_src[block] = block_idx


# Usage: Call these after each stage
# After fine_align:
# has_dups = debug_sentence_alignments(all_sentence_alignments)
#
# After building sent_index:
# debug_sentence_alignment_index(sent_index)
#
# After rebuild.build_paragraph_blocks:
# debug_paragraph_blocks(paragraph_blocks)
#
# After rebuild.build_chunks:
# debug_chunks(chunks)