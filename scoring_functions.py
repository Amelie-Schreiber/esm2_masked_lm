import torch
from transformers import AutoTokenizer, EsmForMaskedLM
import torch.nn.functional as F
from typing import List, Tuple



def masked_marginal_scoring(
    tokenizer: AutoTokenizer,
    model: EsmForMaskedLM,
    sequence: str,
    mutations: List[Tuple[int, str, str]]
) -> float:
    
    # Create a copy of the sequence for mutation
    seq_list = list(sequence)

    # Check and mask the positions
    for pos, wt, mt in mutations:
        if seq_list[pos] != wt:
            raise ValueError(f"The amino acid at position {pos} is {seq_list[pos]}, not {wt}.")
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

    # Initialize score
    score = 0

    # Iterate over each mutation
    for (pos, wt, mt), mask_index in zip(mutations, mask_indices):
        # Get the logits for the masked position
        position_logits = logits[0, mask_index]

        # Apply softmax to logits to get probabilities
        probabilities = torch.nn.functional.softmax(position_logits, dim=-1)

        # Convert probabilities to log probabilities
        log_probabilities = torch.log(probabilities)

        # Get the token ids for wt and mt
        wt_token_id = tokenizer.convert_tokens_to_ids(wt)
        mt_token_id = tokenizer.convert_tokens_to_ids(mt)

        # Retrieve the log probabilities
        wt_log_prob = log_probabilities[wt_token_id].item()
        mt_log_prob = log_probabilities[mt_token_id].item()

        # Compute the difference and update the score
        score += mt_log_prob - wt_log_prob

    return score



def mutant_marginal_score(tokenizer, model, sequence, mutations):
    # Validate the mutations
    for position, wt, mt in mutations:
        assert sequence[position] == wt, f"Expected {wt} at position {position}, but found {sequence[position]}"

    # Generate the mutated sequence
    sequence_list = list(sequence)
    for position, _, mt in mutations:
        sequence_list[position] = mt
    mutated_sequence = ''.join(sequence_list)

    # Tokenize the original and mutated sequences
    inputs = tokenizer(sequence, return_tensors='pt')
    inputs_mutated = tokenizer(mutated_sequence, return_tensors='pt')

    # Compute the logits
    with torch.no_grad():
        outputs = model(**inputs)
        logits_wt = outputs.logits
        outputs_mutated = model(**inputs_mutated)
        logits_mt = outputs_mutated.logits

    # Compute the softmax of the logits
    probabilities_wt = F.softmax(logits_wt, dim=-1)
    probabilities_mt = F.softmax(logits_mt, dim=-1)

    # Compute the log probabilities
    log_probabilities_wt = torch.log(probabilities_wt)
    log_probabilities_mt = torch.log(probabilities_mt)

    # Compute the difference in log probabilities for the mutated positions
    log_probability_difference = 0
    for position, wt, mt in mutations:
        wt_index = tokenizer.convert_tokens_to_ids(wt)
        mt_index = tokenizer.convert_tokens_to_ids(mt)
        log_probability_difference += log_probabilities_mt[0, position, mt_index] - log_probabilities_mt[0, position, wt_index]

    return log_probability_difference.item()



def wild_type_marginal_score(tokenizer, model, sequence, mutations):
    # Verify that the mutations match the sequence
    for pos, wt, mt in mutations:
        if sequence[pos] != wt:
            return f"Error: Position {pos} in sequence is {sequence[pos]}, not {wt}"
    
    # Convert sequence to input_ids and create a copy for mutations
    inputs = tokenizer(sequence, return_tensors="pt")
    inputs_mutated = inputs.input_ids.clone()
    
    # Apply mutations to inputs
    for pos, wt, mt in mutations:
        inputs_mutated[0, pos] = tokenizer.convert_tokens_to_ids(mt)
    
    # Compute logits for both sequences
    with torch.no_grad():
        logits_wt = model(inputs.input_ids).logits
        logits_mt = model(inputs_mutated).logits
    
    # Apply softmax to convert logits to probabilities
    probabilities_wt = torch.nn.functional.softmax(logits_wt, dim=-1)
    probabilities_mt = torch.nn.functional.softmax(logits_mt, dim=-1)
    
    # Compute the log probabilities
    log_probabilities_wt = torch.log(probabilities_wt)
    log_probabilities_mt = torch.log(probabilities_mt)
    
    # Compute the difference in log probabilities for the mutated positions
    log_probability_difference = 0
    for pos, wt, mt in mutations:
        log_probability_difference += (
            log_probabilities_mt[0, pos, tokenizer.convert_tokens_to_ids(mt)] -
            log_probabilities_wt[0, pos, tokenizer.convert_tokens_to_ids(wt)]
        )
    
    return log_probability_difference.item()



def pseudolikelihood_score(tokenizer, model, sequence, mutations):
    # Verify that the mutations match the sequence
    for pos, wt, mt in mutations:
        if sequence[pos] != wt:
            return f"Error: Position {pos} in sequence is {sequence[pos]}, not {wt}"
    
    # Convert sequence to input_ids and create a copy for mutations
    inputs = tokenizer(sequence, return_tensors="pt")
    
    pseudolikelihood_difference = 0.0

    # For each position in the mutations
    for pos, wt, mt in mutations:
        # Create two copies of the original input_ids: one for the wild-type and one for the mutant
        inputs_wt = inputs.input_ids.clone()
        inputs_mt = inputs.input_ids.clone()

        # Replace the token at the current position with the mutant in inputs_mt
        inputs_mt[0, pos] = tokenizer.convert_tokens_to_ids(mt)

        # Compute logits for both sequences
        with torch.no_grad():
            logits_wt = model(inputs_wt).logits
            logits_mt = model(inputs_mt).logits

        # Apply softmax to convert logits to probabilities
        probabilities_wt = torch.nn.functional.softmax(logits_wt, dim=-1)
        probabilities_mt = torch.nn.functional.softmax(logits_mt, dim=-1)

        # Compute the log probabilities
        log_probabilities_wt = torch.log(probabilities_wt)
        log_probabilities_mt = torch.log(probabilities_mt)

        # Compute the difference in log probabilities for the current position
        pseudolikelihood_difference += (
            log_probabilities_mt[0, pos, tokenizer.convert_tokens_to_ids(mt)] -
            log_probabilities_wt[0, pos, tokenizer.convert_tokens_to_ids(wt)]
        )
    
    return pseudolikelihood_difference.item()



