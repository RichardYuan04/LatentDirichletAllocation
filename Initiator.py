def complete_lda_analysis(data, text_column='OR Brief Description'):
    """
    Complete LDA analysis workflow
    """
    print("Starting Complete LDA Analysis")
    print("=" * 60)
    
    # Prepare text data
    texts = data[text_column].dropna().astype(str).tolist()
    print(f"📄 Valid text count: {len(texts)}")
    
    if len(texts) < 10:
        print("Too few texts, at least 10 documents are recommended")
        return None
    
    # Step 1: Find optimal number of topics
    if len(texts) < 50:
        topic_range = range(2, 8)
    elif len(texts) < 200:
        topic_range = range(2, 12)
    else:
        topic_range = range(2, 16)
    
    print(f"Testing topic range: {min(topic_range)}-{max(topic_range)-1}")
    
    topic_numbers, perplexity_scores, coherence_scores, _, _ = find_optimal_topics(
        texts, topic_range
    )
    
    # Plot metrics
    plot_topic_selection_metrics(topic_numbers, perplexity_scores, coherence_scores)
    
    # Get recommendation
    recommended_topics = recommend_topic_number(topic_numbers, perplexity_scores, coherence_scores)
    
    # Step 2: Perform final LDA with optimal topics
    lda_model, vectorizer, doc_term_matrix, feature_names = perform_final_lda(
        texts, recommended_topics
    )
    
    # Step 3: Display topics
    topics_data = display_topics(lda_model, feature_names)
    
    # Step 4: Visualize topics
    visualize_topics(topics_data)
    
    # Step 5: Analyze document-topic relationships
    doc_topic_probs, dominant_topics = analyze_document_topics(
        lda_model, doc_term_matrix, texts, feature_names
    )
    
    return {
        'lda_model': lda_model,
        'vectorizer': vectorizer,
        'topics_data': topics_data,
        'doc_topic_probs': doc_topic_probs,
        'dominant_topics': dominant_topics,
        'recommended_topics': recommended_topics,
        'texts': texts
    }

# Run complete analysis
result = complete_lda_analysis(data)
