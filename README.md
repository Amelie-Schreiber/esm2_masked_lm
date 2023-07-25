# esm2_masked_lm
Various masked LM ideas using EMS-2. 

- [masked_lm_clustering](https://github.com/Amelie-Schreiber/esm2_masked_lm/blob/main/masked_lm_clustering.ipynb) shows how to perform hierarchical clustering of latent embeddings of proteins using the masked protein language model ESM-2. This uses a sequence with (possibly mutiple) masked residues, computes the top `m` most likely and least likely protein sequences conditioned on all positions being masked simultaneously. It then uses persistent homology, DBSCAN, and HDBSCAN (along with $k$-Means and Agglomerative Clustering for comparison) to cluster the sequences. HDBSCAN returns a clustering hierarchy reminiscent of an evolutionary tree for protein sequences generated by the model. 
