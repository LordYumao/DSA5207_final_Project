import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go

# 示例topic关键词和权重
topics = {
    "topic#0": [0.134, 0.096, 0.095, 0.088, 0.067, 0.056, 0.052, 0.044, 0.041, 0.032, 0.025, 0.024, 0.020, 0.020, 0.020],
    "topic#1": [0.099, 0.069, 0.064, 0.061, 0.054, 0.050, 0.047, 0.047, 0.047, 0.041, 0.041, 0.037, 0.034, 0.034, 0.032],
    "topic#2": [0.148, 0.147, 0.092, 0.069, 0.065, 0.061, 0.058, 0.047, 0.047, 0.043, 0.036, 0.026, 0.024, 0.020, 0.014],
    "topic#3": [0.222, 0.114, 0.078, 0.059, 0.059, 0.059, 0.057, 0.049, 0.033, 0.031, 0.030, 0.027, 0.024, 0.020, 0.020],
    "topic#4": [0.180, 0.155, 0.095, 0.072, 0.071, 0.066, 0.048, 0.044, 0.044, 0.031, 0.030, 0.030, 0.025, 0.002, 0.002],
    "topic#5": [0.250, 0.196, 0.084, 0.064, 0.063, 0.061, 0.048, 0.040, 0.029, 0.020, 0.018, 0.002, 0.002, 0.002, 0.002],
    "topic#6": [0.396, 0.051, 0.045, 0.045, 0.038, 0.038, 0.038, 0.037, 0.035, 0.035, 0.030, 0.030, 0.023, 0.016, 0.015],
    "topic#7": [0.242, 0.098, 0.090, 0.078, 0.073, 0.067, 0.056, 0.034, 0.030, 0.024, 0.023, 0.016, 0.015, 0.013, 0.013]
}

df = pd.DataFrame(topics).T
df.columns = [f"keyword{i+1}" for i in range(df.shape[1])]

similarities = []
if len(df) < 2:
    raise ValueError("dataframe must have at least two rows to calculate similarities.")

for i in range(len(df) - 1):
    sim_matrix = cosine_similarity(df.iloc[i].values.reshape(1, -1), df.iloc[i + 1].values.reshape(1, -1))
    similarities.append(sim_matrix[0])

print(f"similarities: {len(similarities)}")

labels = []
sources = []
targets = []
values = []

for i in range(len(df)):
    labels.append(f"T{i+1}")
    for j in range(len(similarities[i])):
        if similarities[i][j] > 0.1:
            sources.append(i)
            targets.append(j + 1)
            values.append(similarities[i][j] * 100)

fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=labels,
        color="blue"
    ),
    link=dict(
        source=sources,
        target=targets,
        value=values,
        hoverinfo='value+percent'
    ))])

fig.update_layout(
    title_text="topic similarity",
    font_size=12,
    width=1600,
    height=900
)

fig.write_html("topic_similarity_sankey.html")
print("topic_similarity_sankey.html saved successfully.")  # 保存成功的提示topic_similarity_sankey.html")