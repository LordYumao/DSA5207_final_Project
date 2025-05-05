import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


df = pd.read_csv('pred_res.csv')

id2label = {
    0: 'admiration',
    1: 'amusement',
    2: 'anger',
    3: 'annoyance',
    4: 'approval',
    5: 'caring',
    6: 'confusion',
    7: 'curiosity',
    8: 'desire',
    9: 'disappointment',
    10: 'disapproval',
    11: 'disgust',
    12: 'embarrassment',
    13: 'excitement',
    14: 'fear',
    15: 'gratitude',
    16: 'joy',
    17: 'love',
    18: 'optimism',
    19: 'realization',
    20: 'sadness',
    21: 'neutral',
    22: 'other'
}

df['pred_label'] = df['prediction'].map(id2label)

print(df[['text', 'prediction', 'label']].head())

if 'pred_label' in df.columns:
    counts = df['pred_label'].value_counts()
    percentages = df['pred_label'].value_counts(normalize=True) * 100

    summary_df = pd.DataFrame({
        'Count': counts,
        'Percentage (%)': percentages.round(2)
    })

    print(summary_df)

    plt.figure(figsize=(10, 5))
    counts.plot(kind='bar')
    plt.xlabel('Predicted Sentiment')
    plt.ylabel('Count')
    plt.title('Distribution of Predicted Sentiments')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

if 'label' in df.columns and 'pred_label' in df.columns:
    df['label'] = df['label'].astype(str).fillna('unknown')
    df['pred_label'] = df['pred_label'].astype(str).fillna('unknown')
    all_labels = sorted(set(df['label']).union(set(df['pred_label'])))

    cm = confusion_matrix(df['label'], df['pred_label'], labels=all_labels)
    cm_df = pd.DataFrame(cm, index=all_labels, columns=all_labels)

    print("\nconfusion_matrix：")
    print(cm_df)

    report_dict = classification_report(df['label'], df['pred_label'], output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()

