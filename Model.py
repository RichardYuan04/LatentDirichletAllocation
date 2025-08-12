# Complete LDA analysis with topic display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import warnings
warnings.filterwarnings('ignore')

def find_optimal_topics(texts, topic_range=range(2, 16), max_features=100):
    """
    Find the optimal number of topics using perplexity and coherence
    """
    
    # Text preprocessing
    vectorizer = CountVectorizer(
        max_features=max_features,
        min_df=2,
        max_df=0.8,
        ngram_range=(1, 2),
        token_pattern=r'(?u)\b\w+\b'
    )
    
    doc_term_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    # Store results
    perplexity_scores = []
    coherence_scores = []
    topic_numbers = []
    
    print("Testing different topic numbers...")
    print("-" * 40)
    
    for n_topics in topic_range:
        print(f"Testing {n_topics} topics...", end="")
        
        # Train LDA model
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=20
        )
        lda.fit(doc_term_matrix)
        
        # Calculate perplexity
        perplexity = lda.perplexity(doc_term_matrix)
        
        # Calculate coherence
        coherence = calculate_coherence_score(lda, feature_names, doc_term_matrix)
        
        perplexity_scores.append(perplexity)
        coherence_scores.append(coherence)
        topic_numbers.append(n_topics)
        
        print(f" Perplexity: {perplexity:.2f}, Coherence: {coherence:.3f}")
    
    return topic_numbers, perplexity_scores, coherence_scores, vectorizer, doc_term_matrix

def calculate_coherence_score(lda_model, feature_names, doc_term_matrix, top_n=10):
    """
    Calculate topic coherence score (safe version)
    """
    coherence_scores = []
    
    # Convert to dense matrix to avoid sparse matrix issues
    dense_matrix = doc_term_matrix.toarray()
    
    for topic_idx, topic in enumerate(lda_model.components_):
        # Get top N words for the topic
        top_words_idx = topic.argsort()[-top_n:][::-1]
        top_words = [feature_names[i] for i in top_words_idx]
        
        # Calculate PMI between word pairs
        word_pairs_pmi = []
        for i in range(len(top_words)):
            for j in range(i+1, len(top_words)):
                word1, word2 = top_words[i], top_words[j]
                
                word1_idx = np.where(feature_names == word1)[0]
                word2_idx = np.where(feature_names == word2)[0]
                
                if len(word1_idx) > 0 and len(word2_idx) > 0:
                    # Use dense matrix for calculation
                    word1_docs = (dense_matrix[:, word1_idx[0]] > 0).sum()
                    word2_docs = (dense_matrix[:, word2_idx[0]] > 0).sum()
                    both_docs = ((dense_matrix[:, word1_idx[0]] > 0) & 
                                (dense_matrix[:, word2_idx[0]] > 0)).sum()
                    
                    if both_docs > 0:
                        pmi = np.log((both_docs * dense_matrix.shape[0]) / 
                                   (word1_docs * word2_docs))
                        word_pairs_pmi.append(pmi)
        
        if word_pairs_pmi:
            coherence_scores.append(np.mean(word_pairs_pmi))
        else:
            coherence_scores.append(0)
    
    return np.mean(coherence_scores)

def plot_topic_selection_metrics(topic_numbers, perplexity_scores, coherence_scores):
    """
    Plot topic selection metrics
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Perplexity plot
    ax1.plot(topic_numbers, perplexity_scores, 'b-o', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Topics')
    ax1.set_ylabel('Perplexity (lower is better)')
    ax1.set_title('Perplexity vs Number of Topics')
    ax1.grid(True, alpha=0.3)
    
    # Mark lowest point
    min_perp_idx = np.argmin(perplexity_scores)
    ax1.annotate(f'Lowest: {topic_numbers[min_perp_idx]}', 
                xy=(topic_numbers[min_perp_idx], perplexity_scores[min_perp_idx]),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Coherence plot
    ax2.plot(topic_numbers, coherence_scores, 'r-o', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Topics')
    ax2.set_ylabel('Coherence (higher is better)')
    ax2.set_title('Coherence vs Number of Topics')
    ax2.grid(True, alpha=0.3)
    
    # Mark highest point
    max_coh_idx = np.argmax(coherence_scores)
    ax2.annotate(f'Highest: {topic_numbers[max_coh_idx]}', 
                xy=(topic_numbers[max_coh_idx], coherence_scores[max_coh_idx]),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    plt.show()
    
    return min_perp_idx, max_coh_idx

def recommend_topic_number(topic_numbers, perplexity_scores, coherence_scores):
    """
    Recommend the best number of topics based on metrics
    """
    print("\n" + "=" * 50)
    print("📊 Topic Number Selection Recommendation")
    print("=" * 50)
    
    # Find extrema
    min_perp_idx = np.argmin(perplexity_scores)
    max_coh_idx = np.argmax(coherence_scores)
    
    min_perp_topics = topic_numbers[min_perp_idx]
    max_coh_topics = topic_numbers[max_coh_idx]
    
    print(f"Lowest Perplexity: {min_perp_topics} topics (Score: {perplexity_scores[min_perp_idx]:.2f})")
    print(f"Highest Coherence: {max_coh_topics} topics (Score: {coherence_scores[max_coh_idx]:.3f})")
    
    # Find elbow point
    perp_diff1 = np.diff(perplexity_scores)
    perp_diff2 = np.diff(perp_diff1)
    
    if len(perp_diff2) > 0:
        elbow_idx = np.argmax(np.abs(perp_diff2)) + 1
        elbow_topics = topic_numbers[elbow_idx]
        print(f"Perplexity Elbow Point: {elbow_topics} topics")
    
    # Comprehensive recommendation
    print(f"\n Recommendation:")
    if min_perp_topics == max_coh_topics:
        print(f"   Both metrics point to {min_perp_topics} topics - highly recommended!")
        recommended = min_perp_topics
    else:
        candidates = [min_perp_topics, max_coh_topics]
        if 'elbow_topics' in locals():
            candidates.append(elbow_topics)
        
        recommended = int(np.median(candidates))
        print(f"   Considering both metrics, suggest trying {min_perp_topics}-{max_coh_topics} topics")
        print(f"   Comprehensive recommendation: {recommended} topics")
    
    return recommended

def perform_final_lda(texts, n_topics, max_features=100):
    """
    Perform final LDA analysis with the optimal number of topics
    """
    print(f"\n Performing final LDA analysis with {n_topics} topics")
    print("=" * 60)
    
    # Text preprocessing
    vectorizer = CountVectorizer(
        max_features=max_features,
        min_df=2,
        max_df=0.8,
        ngram_range=(1, 2),
        token_pattern=r'(?u)\b\w+\b'
    )
    
    doc_term_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    # Train final LDA model
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=50  # More iterations for final model
    )
    lda.fit(doc_term_matrix)
    
    return lda, vectorizer, doc_term_matrix, feature_names

def display_topics(lda_model, feature_names, n_top_words=10):
    """
    Display topic keywords and weights
    """
    print(f"\n Topic Analysis Results:")
    print("=" * 60)
    
    topics_data = []
    
    for topic_idx, topic in enumerate(lda_model.components_):
        top_words_idx = topic.argsort()[-n_top_words:][::-1]
        top_words = [feature_names[i] for i in top_words_idx]
        top_weights = [topic[i] for i in top_words_idx]
        
        print(f"\n🔹 Topic {topic_idx + 1}:")
        print("-" * 30)
        
        topic_words = []
        for word, weight in zip(top_words, top_weights):
            topic_words.append(f"{word} ({weight:.3f})")
            print(f"  {word}: {weight:.4f}")
        
        topics_data.append({
            'topic_id': topic_idx + 1,
            'top_words': top_words,
            'weights': top_weights,
            'word_weight_pairs': topic_words
        })
    
    return topics_data

def analyze_document_topics(lda_model, doc_term_matrix, texts, feature_names, n_examples=5):
    """
    Analyze which documents belong to which topics
    """
    print(f"\n📄 Document-Topic Analysis:")
    print("=" * 60)
    
    # Get document-topic probabilities
    doc_topic_probs = lda_model.transform(doc_term_matrix)
    
    # Find the dominant topic for each document
    dominant_topics = np.argmax(doc_topic_probs, axis=1)
    
    # Show examples for each topic
    for topic_idx in range(lda_model.n_components):
        topic_docs = np.where(dominant_topics == topic_idx)[0]
        
        print(f"\n Topic {topic_idx + 1} - {len(topic_docs)} documents")
        print("-" * 40)
        
        if len(topic_docs) > 0:
            # Show top examples (highest probability for this topic)
            topic_probs_for_topic = doc_topic_probs[topic_docs, topic_idx]
            top_doc_indices = topic_docs[np.argsort(topic_probs_for_topic)[-n_examples:]]
            
            for i, doc_idx in enumerate(reversed(top_doc_indices)):
                prob = doc_topic_probs[doc_idx, topic_idx]
                text_preview = texts[doc_idx][:100] + "..." if len(texts[doc_idx]) > 100 else texts[doc_idx]
                print(f"  Example {i+1} (prob: {prob:.3f}): {text_preview}")
        
    return doc_topic_probs, dominant_topics

def visualize_topics(topics_data):
    """
    Visualize topic word weights
    """
    n_topics = len(topics_data)
    fig, axes = plt.subplots(n_topics, 1, figsize=(12, 4*n_topics))
    
    if n_topics == 1:
        axes = [axes]
    
    for idx, topic_data in enumerate(topics_data):
        words = topic_data['top_words'][:8]  # Top 8 words
        weights = topic_data['weights'][:8]
        
        axes[idx].barh(range(len(words)), weights, color='steelblue')
        axes[idx].set_yticks(range(len(words)))
        axes[idx].set_yticklabels(words)
        axes[idx].set_title(f'Topic {topic_data["topic_id"]} - Top Words')
        axes[idx].set_xlabel('Weight')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

