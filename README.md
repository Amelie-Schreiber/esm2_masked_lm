# esm2_masked_lm
Various masked LM ideas using [EMS-2](https://huggingface.co/docs/transformers/model_doc/esm). 

- [masked_lm_clustering](https://github.com/Amelie-Schreiber/esm2_masked_lm/blob/main/masked_lm_clustering.ipynb) shows how to perform hierarchical clustering of latent embeddings of proteins using the masked protein language model ESM-2. This uses a sequence with (possibly mutiple) masked residues, computes the top `m` most likely and least likely protein sequences conditioned on all positions being masked simultaneously. It then uses persistent homology, DBSCAN, and HDBSCAN (along with $k$-Means and Agglomerative Clustering for comparison) to cluster the sequences. HDBSCAN returns a clustering hierarchy reminiscent of an evolutionary tree for protein sequences generated by the model.

- [ems2_mutations](https://github.com/Amelie-Schreiber/esm2_masked_lm/blob/main/ems2_mutations.ipynb) implements part of the paper [Language models enable zero-shot prediction of the effects of mutations on protein function](https://www.biorxiv.org/content/10.1101/2021.07.09.450648v2) using ESM-2 instead of ESM-1v. See also [the META repo](https://github.com/facebookresearch/esm/tree/main/examples/variant-prediction)

- [scoring_mutations](https://github.com/Amelie-Schreiber/esm2_masked_lm/blob/main/scoring_mutations.ipynb) computes the `masked_marginal_score`, the `wild_type_marginal_score`, the `mutant_type_marginal_score`, and the `pseudolikelihood_score` for a list of mutated sequences predicted to be the most and least likely by ESM-2 based on a fixed wild-type sequences, and with a fixed target mutation sequence. This is closely related to the previous notebook, and finishes implementing the scoring functions mentioned in [Language models enable zero-shot prediction of the effects of mutations on protein function](https://www.biorxiv.org/content/10.1101/2021.07.09.450648v2) using ESM-2. You can swap out `facebook/esm2_t6_8M_UR50D` for one of the [other larger models](https://huggingface.co/facebook/esm2_t6_8M_UR50D).
  
- [sequence_classification](https://github.com/Amelie-Schreiber/esm2_masked_lm/blob/main/sequence_classification.ipynb) builds a basic protein sequence classifier with three labels for enzymes, receptor proteins, and structural proteins. It uses the `facebook/esm2_t6_8M_UR50D` and thus is lightweight and easy to train, yet accurate. 
