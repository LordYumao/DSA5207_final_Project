import re
from datetime import datetime
import pandas as pd
from gensim import corpora, models
import pyLDAvis.gensim_models
from collections import defaultdict

# ===== Configuration =====
INPUT_FILE = "comments_translated.xlsx"    # Path to input Excel file
TEXT_COLUMN = "weibo_text"                 # Name of the text column
N_TOPICS = 8                               # Number of topics to generate
MIN_WORD_FREQ = 5                          # Minimum word frequency threshold
MAX_WORD_FREQ_RATIO = 0.5                  # Maximum word frequency ratio (50%)

# ===== Load data =====
df = pd.read_excel(INPUT_FILE)
texts = df[TEXT_COLUMN].dropna().tolist()

# ===== Data preprocessing =====
def preprocess_chinese(text: str) -> str:
    """
    Remove non-Chinese characters and return cleaned text.
    """
    # Keep only Chinese characters
    cleaned = re.sub(r'[^\u4e00-\u9fa5]+', '', text)
    return cleaned

# Apply preprocessing to all documents
processed_docs = [preprocess_chinese(t) for t in texts]

# ===== Build word frequency map =====
frequency = defaultdict(int)
for doc in processed_docs:
    for char in doc:
        frequency[char] += 1  # here each Chinese character is treated as a token

# Compute dynamic maximum frequency threshold
max_freq_threshold = int(len(processed_docs) * MAX_WORD_FREQ_RATIO)

# Filter out rare and overly frequent tokens
filtered_docs = [
    [char for char in doc
     if MIN_WORD_FREQ <= frequency[char] <= max_freq_threshold]
    for doc in processed_docs
]

# ===== Create dictionary and corpus =====
dictionary = corpora.Dictionary(filtered_docs)
corpus = [dictionary.doc2bow(doc) for doc in filtered_docs]

# ===== Train LDA model =====
lda_model = models.LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=N_TOPICS,
    passes=20,            # increase number of passes for stability
    alpha='asymmetric',   # asymmetric prior for document-topic distribution
    eta='auto',           # learn topic-word distributions automatically
    chunksize=2000,
    eval_every=5
)

# ===== Generate HTML report =====
def generate_html_report(zh_topics, en_topics, filename: str):
    """Generate a bilingual (Chinese + English) HTML report of topics."""
    html_template = """
    <html>
      <head>
        <meta charset="utf-8">
        <title>LDA Topic Analysis Report</title>
        <style>
          body {{ font-family: Arial, sans-serif; margin: 2em; }}
          .topic {{ border: 1px solid #e0e0e0; border-radius: 8px; padding: 1.5em; margin-bottom: 1.5em; }}
          h1 {{ color: #2c3e50; }}
          h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 0.3em; }}
          .timestamp {{ color: #95a5a6; margin-bottom: 2em; }}
          .en {{ color: #2c3e50; font-weight: 500; margin-top: 0.5em; }}
          .zh {{ color: #7f8c8d; margin-top: 0.3em; font-style: italic; }}
        </style>
      </head>
      <body>
        <h1>Weibo Comments Topic Analysis Report</h1>
        <div class="timestamp">Generated on: {timestamp}</div>
        {topic_blocks}
      </body>
    </html>
    """

    topic_blocks = []
    for idx, (zh_words, en_words) in enumerate(zip(zh_topics, en_topics)):
        block = f"""
        <div class="topic">
          <h2>Topic #{idx + 1}</h2>
          <div class="en"><strong>Core Issues:</strong> {', '.join(en_words)}</div>
          <div class="zh"><strong>Original Keywords:</strong> {' | '.join(zh_words)}</div>
        </div>
        """
        topic_blocks.append(block)

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_template.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            topic_blocks='\n'.join(topic_blocks)
        ))

# ===== Prepare topic keywords =====
topics_zh = []
topics_en = []
for topic_id in range(N_TOPICS):
    topic_terms = lda_model.show_topic(topic_id, topn=15)
    zh_list = [term for term, _ in topic_terms]
    en_list = [f"{term}({weight:.3f})" for term, weight in topic_terms]
    topics_zh.append(zh_list)
    topics_en.append(en_list)

# Generate the HTML report
generate_html_report(topics_zh, topics_en, "weibo_topic_analysis.html")
print("Report generated: weibo_topic_analysis.html")

# ===== Visualize topics with pyLDAvis =====
vis_data = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
pyLDAvis.save_html(vis_data, 'topic_visualization.html')
print("Visualization saved as topic_visualization.html")
