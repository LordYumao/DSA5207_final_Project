import sys
import pandas as pd
import gensim
from gensim import corpora
from gensim.models import LdaModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis


def display_topics(lda_model, no_top_words):
    """Print the top words for each topic."""
    for idx in range(lda_model.num_topics):
        terms = lda_model.show_topic(idx, topn=no_top_words)
        term_list = [term for term, weight in terms]
        print(f"Topic {idx+1}: {' '.join(term_list)}")


def main(input_file: str,
         text_column: str = None,
         n_topics: int = 10,
         no_top_words: int = 10):
    # 1. Load data
    try:
        df = pd.read_excel(input_file)
    except Exception as e:
        print(f"File read error: {e}")
        sys.exit(1)

    # 2. Determine which column to use for text
    candidates = []
    if text_column:
        candidates.append(text_column)
    # fallback column names
    candidates.extend(['text_en', 'weibo_text_en', 'weibo_text'])
    selected = next((col for col in candidates if col in df.columns), None)
    if not selected:
        print(f"Error: No text column found. Available columns: {list(df.columns)}")
        sys.exit(1)
    if text_column and selected != text_column:
        print(f"Note: Using column '{selected}' for topic analysis.")

    # 3. Extract documents
    docs = df[selected].astype(str).tolist()

    # 4. Simple tokenization (split on whitespace, lowercase)
    texts = [doc.lower().split() for doc in docs]

    # 5. Build dictionary and corpus
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # 6. Train the LDA model
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=n_topics,
        random_state=42,
        passes=10
    )

    # 7. Print out topic keywords
    print("\n=== LDA Topic Keywords ===")
    display_topics(lda_model, no_top_words)

    # 8. Visualize with pyLDAvis and save as HTML
    print("\nGenerating pyLDAvis visualization (gensim) and saving to lda_vis.html ...")
    vis_data = gensimvis.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(vis_data, 'lda_vis.html')
    print("Done! Visualization saved as lda_vis.html")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='LDA Topic Modeling Script (gensim)'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='comments_translated.xlsx',
        help='Input Excel file containing the text column for analysis'
    )
    parser.add_argument(
        '--column', '-c',
        type=str,
        default=None,
        help='Optional: specify the name of the text column'
    )
    parser.add_argument(
        '--topics', '-t',
        type=int,
        default=10,
        help='Number of topics (default: 10)'
    )
    parser.add_argument(
        '--top_words',
        type=int,
        default=10,
        help='Number of top words to display per topic'
    )
    args = parser.parse_args()

    main(
        input_file=args.input,
        text_column=args.column,
        n_topics=args.topics,
        no_top_words=args.top_words
    )
