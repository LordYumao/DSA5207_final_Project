import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv('pred_res.csv')
print("dataframe：")
print(df.head())

if 'prediction' in df.columns:
    counts = df['prediction'].value_counts()
    percentages = df['prediction'].value_counts(normalize=True) * 100

    summary_df = pd.DataFrame({
        'Count': counts,
        'Percentage (%)': percentages.round(2)
    })

    print("\nprediction of sentiment：")
    print(summary_df)

    plt.figure(figsize=(6, 4))
    counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.xlabel('Predicted Sentiment')
    plt.ylabel('Count')
    plt.title('Distribution of Predicted Sentiments')
    plt.tight_layout()
    plt.show()

if 'label' in df.columns and 'prediction' in df.columns:
    labels = sorted(df['label'].unique())
    cm = confusion_matrix(df['label'], df['prediction'], labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    print("\nconfusion matrix：")
    print(cm_df)

    report_dict = classification_report(df['label'], df['prediction'], output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()

    print("\nclassification report：")
    print(report_df.round(2))
