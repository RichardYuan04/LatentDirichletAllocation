# MTR Accident Text Analysis - LDA Topic Modeling
## Project Overview
This project was completed during an internship at MTR Corporation, focusing on accident text analysis using Latent Dirichlet Allocation (LDA) topic modeling techniques. The project performs in-depth analysis of accident reports to discover accident patterns, identify key risk factors, and provide data-driven insights for MTR's safety management.

## Background
As Hong Kong's essential public transportation system, MTR serves millions of passengers daily. Accident prevention and safety management are paramount to operations. Through analyzing historical accident reports, we can:

+ Identify common accident types and patterns
+ Discover potential risk factors
+ Provide evidence for prevention strategy development
+ Optimize safety management processes
+ Technical Methodology
+ LDA Topic Modeling Principles

LDA is an unsupervised machine learning algorithm particularly suitable for document topic discovery. In accident analysis, LDA can:

* Automatically discover hidden topics: Identify different accident types and patterns from large volumes of accident reports
* Probabilistic distribution modeling: Each document (accident report) contains probability distributions over multiple topics
* Keyword extraction: Generate most relevant keywords for each topic to facilitate understanding of accident characteristics

Core Functions
1. `find_optimal_topics()`
Determines the optimal number of topics using perplexity and coherence metrics.

2. `calculate_coherence_score()`
Calculates topic coherence score using Point-wise Mutual Information (PMI) between word pairs.

3. `plot_topic_selection_metrics()`
Visualizes perplexity and coherence scores across different topic numbers to aid in optimal topic selection.

4. `recommend_topic_number()`
Provides intelligent recommendations for the best number of topics based on multiple evaluation metrics.

5. `perform_final_lda()`
Trains the final LDA model using the optimal number of topics with enhanced parameters.

6. `display_topics()`
Displays topic keywords and their weights in a structured format.

7. `analyze_document_topics()`
Analyzes document-topic relationships and provides representative examples for each topic.

8. `visualize_topics()`
Creates horizontal bar charts showing keyword weights for each topic.

9. `complete_lda_analysis()`
Orchestrates the entire LDA analysis workflow from data preparation to final results.
