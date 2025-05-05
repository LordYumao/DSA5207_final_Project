import sys
import pandas as pd
from gensim import corpora
from gensim.models import LdaModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import jieba
import re
import argparse


def chinese_preprocessor(text: str) -> list:
    """Perform Chinese text preprocessing: remove non-Chinese characters and tokenize."""
    # Remove any character that is not a Chinese character
    cleaned = re.sub(r'[^\u4e00-\u9fa5]+', '', text)
    # Precise segmentation with jieba
    tokens = jieba.lcut(cleaned)
    # Keep tokens longer than one character
    return [tok for tok in tokens if len(tok) > 1]


def display_topics(lda_model: LdaModel, no_top_words: int):
    """Print out the top words for each topic."""
    print("\n=== LDA Topic Keywords ===")
    for idx in range(lda_model.num_topics):
        terms = lda_model.show_topic(idx, topn=no_top_words)
        term_list = [f"{term}({weight:.3f})" for term, weight in terms]
        print(f"Topic #{idx}: {' | '.join(term_list)}")


def main(input_file: str,
         text_column: str = None,
         n_topics: int = 8,
         no_top_words: int = 15):

    # 1. Read data
    try:
        df = pd.read_excel(input_file)
    except Exception as e:
        print(f"File read error: {e}")
        sys.exit(1)

    # 2. Determine which column contains the text
    candidates = ['weibo_text', 'text_en', 'weibo_text_en'] + ([text_column] if text_column else [])
    selected = next((col for col in candidates if col in df.columns), None)
    if not selected:
        print(f"Error: No text column found. Available columns: {list(df.columns)}")
        sys.exit(1)

    # 3. Preprocess texts
    print("\n=== Preprocessing texts ===")
    texts = []
    for doc in df[selected].astype(str):
        tokens = chinese_preprocessor(doc)
        texts.append(tokens)

    # 4. Build dictionary and corpus
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(doc) for doc in texts]

    # 5. Train LDA model
    print("\n=== Training LDA model ===")
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=n_topics,
        random_state=42,
        passes=20,
        alpha='auto'
    )

    # 6. Display topic keywords
    display_topics(lda_model, no_top_words)

    # 7. Generate interactive visualization
    print("\nGenerating pyLDAvis visualization...")
    vis_data = gensimvis.prepare(
        lda_model,
        corpus,
        dictionary,
        sort_topics=False,  # preserve original topic order
        mds='mmds',         # use metric multidimensional scaling
        R=30                # show 30 terms in the chart
    )
    pyLDAvis.save_html(vis_data, 'topic_visualization_single.html')
    print("Done! Visualization saved as topic_visualization_single.html")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Weibo Comments LDA Topic Analysis')
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to input Excel file'
    )
    parser.add_argument(
        '--column', '-c',
        help='Name of the text column (default: auto-detect)'
    )
    parser.add_argument(
        '--topics', '-t',
        type=int,
        default=8,
        help='Number of topics to generate'
    )
    parser.add_argument(
        '--top_words',
        type=int,
        default=15,
        help='Number of top words to display per topic'
    )
    args = parser.parse_args()

    main(
        input_file=args.input,
        text_column=args.column,
        n_topics=args.topics,
        no_top_words=args.top_words
    )
