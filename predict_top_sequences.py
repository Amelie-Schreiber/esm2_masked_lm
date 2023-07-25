
import torch
from transformers import AutoTokenizer, EsmForMaskedLM
from itertools import product
import heapq



def predict_top_full_sequences(sequence: str, mask_positions: list, m: int):
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model = EsmForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")

    # Create a copy of the sequence for mutation
    seq_list = list(sequence)

    # Mask the positions
    for pos in mask_positions:
        seq_list[pos] = tokenizer.mask_token

    # Convert the mutated sequence back to string
    masked_sequence = "".join(seq_list)
    
    # Tokenize the masked sequence
    inputs = tokenizer(masked_sequence, return_tensors="pt")

    # Get the model's output
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the logits
    logits = outputs.logits

    # Get the mask token indices
    mask_indices = torch.where(inputs.input_ids.squeeze() == tokenizer.mask_token_id)[0]

    # Get the logits for the masked positions
    masked_logits = logits[0, mask_indices]

    # Apply softmax to logits to get probabilities
    masked_probs = torch.nn.functional.softmax(masked_logits, dim=-1)

    # Create a list to store the top sequences and their scores
    top_sequences = []

    # Get all possible combinations of amino acids for the masked positions
    amino_acids = tokenizer.get_vocab().keys()
    combinations = list(product(amino_acids, repeat=len(mask_positions)))

    for combination in combinations:
        # Compute the sum of the log probabilities for this combination
        score = sum(torch.log(masked_probs[i, tokenizer.convert_tokens_to_ids(a)]) for i, a in enumerate(combination))

        # Update the list of top sequences
        if len(top_sequences) < m:
            # If there's room, just add the current sequence
            heapq.heappush(top_sequences, (score.item(), combination))
        else:
            # If there's no room, replace the lowest-scoring sequence if the current sequence is better
            heapq.heappushpop(top_sequences, (score.item(), combination))

    # Create the full sequences by replacing the masked positions with the predicted amino acids
    scores, top_sequences = zip(*top_sequences)
    top_full_sequences = []
    for seq in top_sequences:
        full_seq_list = list(sequence)
        for pos, aa in zip(mask_positions, seq):
            full_seq_list[pos] = aa
        top_full_sequences.append("".join(full_seq_list))

    return scores, top_full_sequences



# Function to predict the bottom m sequences after masking (No Special Tokens)
def predict_bottom_full_sequences_nst(sequence: str, mask_positions: list, m: int):
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model = EsmForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")

    # Create a copy of the sequence for mutation
    seq_list = list(sequence)

    # Mask the positions
    for pos in mask_positions:
        seq_list[pos] = tokenizer.mask_token

    # Convert the mutated sequence back to string
    masked_sequence = "".join(seq_list)
    
    # Tokenize the masked sequence
    inputs = tokenizer(masked_sequence, return_tensors="pt")

    # Get the model's output
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the logits
    logits = outputs.logits

    # Get the mask token indices
    mask_indices = torch.where(inputs.input_ids.squeeze() == tokenizer.mask_token_id)[0]

    # Get the logits for the masked positions
    masked_logits = logits[0, mask_indices]

    # Apply softmax to logits to get probabilities
    masked_probs = torch.nn.functional.softmax(masked_logits, dim=-1)

    # Create a list to store the bottom sequences and their scores
    bottom_sequences = []

    # Get all possible combinations of amino acids for the masked positions
    amino_acids = tokenizer.get_vocab().keys()
    combinations = list(product(amino_acids, repeat=len(mask_positions)))

    for combination in combinations:
        # Exclude sequences with special tokens (No Special Tokens)
        if any(aa in ['<cls>', '<pad>', '<eos>', '<unk>', '.', '-', '<null_1>', '<mask>', 'X', 'B', 'U', 'Z', 'O'] for aa in combination):
            continue

        # Compute the sum of the log probabilities for this combination
        score = sum(torch.log(masked_probs[i, tokenizer.convert_tokens_to_ids(a)]) for i, a in enumerate(combination))

        # Update the list of bottom sequences
        if len(bottom_sequences) < m:
            # If there's room, just add the current sequence
            heapq.heappush(bottom_sequences, (-score.item(), combination))
        else:
            # If there's no room, replace the highest-scoring sequence if the current sequence is worse
            heapq.heappushpop(bottom_sequences, (-score.item(), combination))

    # Create the full sequences by replacing the masked positions with the predicted amino acids
    neg_scores, bottom_sequences = zip(*bottom_sequences)
    scores = [-1 * neg_score for neg_score in neg_scores]
    bottom_full_sequences = []
    for seq in bottom_sequences:
        full_seq_list = list(sequence)
        for pos, aa in zip(mask_positions, seq):
            full_seq_list[pos] = aa
        bottom_full_sequences.append("".join(full_seq_list))

    return scores, bottom_full_sequences



# Includes special tokens
def predict_bottom_full_sequences_st(sequence: str, mask_positions: list, m: int):
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model = EsmForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")

    # Create a copy of the sequence for mutation
    seq_list = list(sequence)

    # Mask the positions
    for pos in mask_positions:
        seq_list[pos] = tokenizer.mask_token

    # Convert the mutated sequence back to string
    masked_sequence = "".join(seq_list)
    
    # Tokenize the masked sequence
    inputs = tokenizer(masked_sequence, return_tensors="pt")

    # Get the model's output
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the logits
    logits = outputs.logits

    # Get the mask token indices
    mask_indices = torch.where(inputs.input_ids.squeeze() == tokenizer.mask_token_id)[0]

    # Get the logits for the masked positions
    masked_logits = logits[0, mask_indices]

    # Apply softmax to logits to get probabilities
    masked_probs = torch.nn.functional.softmax(masked_logits, dim=-1)

    # Create a list to store the bottom sequences and their scores
    bottom_sequences = []

    # Get all possible combinations of amino acids for the masked positions
    amino_acids = tokenizer.get_vocab().keys()
    combinations = list(product(amino_acids, repeat=len(mask_positions)))

    for combination in combinations:
        # Compute the sum of the log probabilities for this combination
        score = sum(torch.log(masked_probs[i, tokenizer.convert_tokens_to_ids(a)]) for i, a in enumerate(combination))

        # Update the list of bottom sequences
        if len(bottom_sequences) < m:
            # If there's room, just add the current sequence
            heapq.heappush(bottom_sequences, (-score.item(), combination))
        else:
            # If there's no room, replace the highest-scoring sequence if the current sequence is worse
            heapq.heappushpop(bottom_sequences, (-score.item(), combination))

    # Create the full sequences by replacing the masked positions with the predicted amino acids
    neg_scores, bottom_sequences = zip(*bottom_sequences)
    scores = [-1 * neg_score for neg_score in neg_scores]
    bottom_full_sequences = []
    for seq in bottom_sequences:
        full_seq_list = list(sequence)
        for pos, aa in zip(mask_positions, seq):
            full_seq_list[pos] = aa
        bottom_full_sequences.append("".join(full_seq_list))

    return scores, bottom_full_sequences