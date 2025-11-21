import textwrap

def count_words(text):
    return len(text.split())


def build_paragraph_blocks(all_paragraph_alignments_sorted,
                           sentence_alignment_index):
    """
    Turn each aligned paragraph pair/group into a (src_block, tgt_block, para_ids) tuple.
    sentence_alignment_index: dict keyed by (paraA_indices_tuple, paraB_indices_tuple)
        giving the list of sentence-level alignments.
    """
    blocks = []
    for entry in all_paragraph_alignments_sorted:
        paraA_idx = entry["src_idx"]  # This is now a tuple like (0, 1)
        paraB_idx = entry["tgt_idx"]  # This is now a tuple like (0,)
        
        # Ensure they're tuples (in case single values were passed)
        if not isinstance(paraA_idx, tuple):
            paraA_idx = (paraA_idx,)
        if not isinstance(paraB_idx, tuple):
            paraB_idx = (paraB_idx,)
        
        sent_pairs = sentence_alignment_index.get((paraA_idx, paraB_idx), [])
        
        if not sent_pairs:
            # No aligned sentences – skip
            continue
        
        # Reconstruct paragraph-level text
        src_sents = [p["src_text"] for p in sent_pairs]
        tgt_sents = [p["tgt_text"] for p in sent_pairs]
        src_block = " ".join(src_sents)
        tgt_block = " ".join(tgt_sents)
        
        blocks.append({
            "src_idx": paraA_idx,  # Keep as tuple
            "tgt_idx": paraB_idx,  # Keep as tuple
            "src_block": src_block,
            "tgt_block": tgt_block,
            "src_words": count_words(src_block),
            "tgt_words": count_words(tgt_block),
        })
    
    return blocks


def build_chunks(paragraph_blocks, max_words=2500):
    """
    Group paragraph blocks into chunks not exceeding max_words
    on either language side.
    """
    chunks = []

    current_src = []
    current_tgt = []
    current_para_ids = []

    current_src_words = 0
    current_tgt_words = 0

    for block in paragraph_blocks:
        b_src = block["src_block"]
        b_tgt = block["tgt_block"]
        w_src = block["src_words"]
        w_tgt = block["tgt_words"]

        # Would adding this block exceed?
        if (current_src_words + w_src > max_words) or \
           (current_tgt_words + w_tgt > max_words):

            # Flush current chunk
            if current_para_ids:
                chunks.append({
                    "paragraph_pairs": current_para_ids,
                    "src_text": "\n\n".join(current_src),
                    "tgt_text": "\n\n".join(current_tgt),
                    "src_words": current_src_words,
                    "tgt_words": current_tgt_words,
                })

            # Start a new chunk
            current_src = [b_src]
            current_tgt = [b_tgt]
            current_para_ids = [(block["src_idx"], block["tgt_idx"])]
            current_src_words = w_src
            current_tgt_words = w_tgt

        else:
            # Add block to current chunk
            current_src.append(b_src)
            current_tgt.append(b_tgt)
            current_para_ids.append((block["src_idx"], block["tgt_idx"]))
            current_src_words += w_src
            current_tgt_words += w_tgt

    # Flush final chunk
    if current_para_ids:
        chunks.append({
            "paragraph_pairs": current_para_ids,
            "src_text": "\n\n".join(current_src),
            "tgt_text": "\n\n".join(current_tgt),
            "src_words": current_src_words,
            "tgt_words": current_tgt_words,
        })

    return chunks
