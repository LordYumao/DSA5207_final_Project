import re
import jieba
import pandas as pd
from gensim import corpora, models
import pyLDAvis.gensim_models
from collections import defaultdict

# ===== Configuration =====
INPUT_FILE = "comments_translated.xlsx"   # Path to input Excel file
TEXT_COLUMN = "weibo_text"                # Name of the text column
CUSTOM_DICT = "custom_dict.txt"           # Path to user dictionary for jieba
STOPWORDS_FILE = "stopwords.txt"          # Path to stopwords file
N_TOPICS = 8                              # Number of topics to generate
MIN_WORD_FREQ = 5                         # Minimum word frequency threshold
MAX_WORD_FREQ_RATIO = 0.5                 # Maximum word frequency ratio (50%)

# ===== Load user dictionary =====
jieba.load_userdict(CUSTOM_DICT)

# ===== Load stopwords =====
with open(STOPWORDS_FILE, 'r', encoding='utf-8') as f:
    stopwords = set(line.strip() for line in f)

# ===== Data loading and preprocessing =====
df = pd.read_excel(INPUT_FILE)
raw_texts = df[TEXT_COLUMN].dropna().tolist()

def preprocess_chinese(text):
    """
    1) Remove non-Chinese characters
    2) Perform precise segmentation with jieba
    3) Filter out stopwords, single characters, and pure digits
    """
    # Remove any character that is not a Chinese character
    cleaned = re.sub(r'[^\u4e00-\u9fa5]+', '', text)
    # Tokenize
    tokens = jieba.lcut(cleaned)
    # Filter tokens
    return [
        tok for tok in tokens
        if len(tok) > 1
        and tok not in stopwords
        and not tok.isdigit()
    ]

# Apply preprocessing to all documents
processed_docs = [preprocess_chinese(doc) for doc in raw_texts]

# ===== Build word frequency map =====
frequency = defaultdict(int)
for doc in processed_docs:
    for word in doc:
        frequency[word] += 1

# Compute dynamic maximum frequency threshold
max_freq_threshold = int(len(processed_docs) * MAX_WORD_FREQ_RATIO)

# Filter tokens by frequency thresholds
processed_docs = [
    [
        word for word in doc
        if MIN_WORD_FREQ <= frequency[word] <= max_freq_threshold
    ]
    for doc in processed_docs
]

# ===== Create dictionary and corpus =====
dictionary = corpora.Dictionary(processed_docs)
corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

# ===== Train LDA model =====
lda_model = models.LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=N_TOPICS,
    passes=20,            # number of training passes
    alpha='asymmetric',   # asymmetric prior for document-topic distribution
    eta='auto',           # learn topic-word distributions automatically
    chunksize=2000,
    eval_every=5
)

# ===== Visualize topics =====
vis_data = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
pyLDAvis.save_html(vis_data, 'topic_visualization.html')

# ===== Print top words for each topic =====
for topic_id in range(N_TOPICS):
    top_terms = lda_model.show_topic(topic_id, topn=15)
    formatted = [f"{term}({weight:.3f})" for term, weight in top_terms]
    print(f"Topic #{topic_id}: " + " | ".join(formatted) + "\n")

print("Visualization saved as topic_visualization.html")
