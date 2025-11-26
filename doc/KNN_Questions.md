# A. Conceptual Fundamentals (Core)

1. What is KNN and how does it work?

→ Tests basic understanding of lazy learning & instance-based models.

2. Is KNN a supervised or unsupervised algorithm? Why?

→ Supervised for classification/regression, but similarity search part is unsupervised.

3. Why is KNN called a “lazy learner”?

→ No training; all computation deferred to inference.

4. What kinds of problems can KNN solve?

→ Classification, regression, density estimation, outlier detection, recommender systems.

5. What is the bias–variance trade-off in KNN?

→ Small K → high variance; large K → high bias.


⸻
# B. Mathematics & Distance Metrics

6. What distance metrics can be used in KNN?

→ Euclidean, Manhattan, Minkowski, Cosine, Hamming, Mahalanobis.

7. Why choose cosine distance over Euclidean in recommenders?

→ Scale invariant, handles sparsity, focuses on direction not magnitude.

8. How does the curse of dimensionality affect KNN?

→ Distances become indistinguishable; nearest ≈ farthest → performance collapses.

9. How do you handle categorical variables in KNN?

→ One-hot encode, embeddings, or Hamming distance.

10. How do you choose K in KNN?

→ Cross-validation; odd K for binary classification.

⸻

# C. Implementation & Complexity

11. What is the training complexity of KNN?

→ O(1) — only stores data.

12. What is the inference complexity of KNN? Why is it slow?

→ O(N × d) per query (brute-force scanning of entire dataset).

13. How can we speed up KNN for large datasets?

→ KD-tree, ball tree, HNSW, FAISS, LSH (ANN), product quantization.

14. How do you choose between KD-tree and Ball Tree?

→ KD-tree: low dimensions (<20).
→ Ball Tree: works better with many irrelevant features.

15. How would you implement KNN from scratch?

→ Distance computation → sort → vote → return class.

16. What is weighted KNN?

→ Closer neighbors get higher weight.
→ Weight = 1/(distance + ε).

⸻

# D. Practical Engineering Use Cases

17. How would you use KNN in a recommender system?

→ Item-to-item similarity for Amazon-style CF.

18. Why is KNN good for cold-start item recommendations?

→ Immediate similarity from metadata or content.

19. How would you use KNN in search ranking?

→ Vector embeddings → nearest neighbors → candidate retrieval.

20. How is KNN used in anomaly detection?

→ Distance to k-th nearest neighbor → threshold → anomaly score.

21. Is KNN used in production at large scale?

→ Not brute-force; ANN approximations (FAISS, Annoy, HNSW) are used.


⸻
# E. Troubleshooting / Edge Cases

22. What happens if features are not scaled in KNN?

→ Distance dominated by features with large magnitude → wrong neighbors.

23. What if K is too small or too large?

→ K=1 overfits. Large K oversmooths / loses signal.

24. What if the dataset contains many irrelevant features?

→ Distance becomes noisy → apply PCA, feature selection.

25. What if your dataset contains extreme class imbalance?

→ KNN will bias toward majority class. Use distance weighting or SMOTE.

26. What if two classes overlap heavily?

→ KNN struggles → need more expressive model.

⸻

# F. Advanced / Research-Level Questions

27. How does approximate nearest neighbor (ANN) differ from exact KNN?

→ Returns approximate neighbors to gain massive speed-ups.

28. Explain LSH (Locality-Sensitive Hashing).

→ Hashing that preserves locality → similar vectors go to same bucket.

29. Explain HNSW (Hierarchical Navigable Small World Graph).

→ Graph-based ANN search used in FAISS/Weaviate → O(log N) search.

30. What is the effect of metric learning on KNN?

→ Learn a distance metric that improves neighborhood quality.

31. Why is Euclidean distance unsuitable in high dimensions?

→ Concentration phenomenon → all points appear equally distant.

32. How would you use KNN with embeddings from BERT?

→ For semantic search: embed → ANN index → top-k neighbors.


⸻
# G. Behavioral / System Design for Recommender Systems (KNN-specific)

33. Design an Amazon-style item-to-item recommender using KNN.

→ User–item matrix → item vectors → cosine KNN → top-K neighbors → aggregate per user.

34. Explain how Amazon’s 2003 paper used KNN for item similarity.

→ Item-to-item CF with precomputed neighbors.

35. How would you deploy a KNN-based recommender with <10 ms latency?

→ Precomputed neighbors in Redis → ANN for dynamic queries.

36. How do you keep KNN recommendations fresh in real-time?

→ Incremental updates → nightly batch refresh → streaming updates for new items.

37. How do you handle new items (cold start)?