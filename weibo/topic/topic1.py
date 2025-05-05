import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity

# Topic weights for two time slices
time_slices = {
    "2024Q1": {
        0: {"Consumption": 0.134, "Laohu": 0.096, "Yunbei": 0.095, "StockMarket": 0.088},
        1: {"MoneyDistribution": 0.099, "Income": 0.069, "Growth": 0.064, "Improvement": 0.061},
        2: {"Employment": 0.148, "IncomeIncrease": 0.147, "We": 0.092, "Fundamental": 0.069},
        3: {"Raise": 0.222, "Salary": 0.114, "Daily": 0.078, "Cannot": 0.059},
        4: {"Distribution": 0.180, "Support": 0.155, "Laohu": 0.095, "Self": 0.072},
        5: {"Now": 0.250, "Upward": 0.196, "Income": 0.084, "Key": 0.064},
        6: {"Consumption": 0.396, "Stimulus": 0.051, "Vitality": 0.045, "Promotion": 0.045},
        7: {"Consumption": 0.242, "WithMoney": 0.098, "Income": 0.090, "Capability": 0.078}
    },
    "2024Q2": {
        0: {"Consumption": 0.145, "Stocks": 0.102, "Market": 0.088, "Policy": 0.075},
        1: {"TaxCut": 0.121, "Enterprises": 0.098, "Welfare": 0.085, "Stimulus": 0.072},
        2: {"Employment": 0.162, "Guarantee": 0.132, "Training": 0.095, "Policy": 0.082},
        3: {"Salary": 0.231, "Adjustment": 0.155, "Standard": 0.102, "Inflation": 0.088},
        4: {"Distribution": 0.192, "Fairness": 0.145, "Reform": 0.122, "Mechanism": 0.095},
        5: {"Economy": 0.275, "Recovery": 0.188, "Indicator": 0.155, "Expectation": 0.132},
        6: {"DomesticDemand": 0.302, "StimulusMeasures": 0.185, "Measures": 0.165, "Vouchers": 0.145},
        7: {"Savings": 0.258, "Investment": 0.205, "WealthManagement": 0.185, "Risk": 0.155}
    }
}


class TopicAnalyzer:
    def __init__(self, time_slices):
        self.time_slices = time_slices
        self.all_terms = self._get_all_terms()
        self.similarity_matrix = None

    def _get_all_terms(self):
        """Get the full vocabulary across all time slices."""
        terms = set()
        for period in self.time_slices.values():
            for topic in period.values():
                terms.update(topic.keys())
        return sorted(terms)

    def _topic_to_vector(self, topic_dict):
        """Convert a single topic (term→weight) into a dense vector."""
        vector = np.zeros(len(self.all_terms))
        for term, weight in topic_dict.items():
            idx = self.all_terms.index(term)
            vector[idx] = weight
        return vector

    def calculate_similarity(self):
        """Calculate cosine similarity between topics of consecutive time slices."""
        periods = list(self.time_slices.keys())
        prev_topics = list(self.time_slices[periods[0]].values())
        next_topics = list(self.time_slices[periods[1]].values())

        # Convert topics to vectors
        prev_vectors = np.array([self._topic_to_vector(t) for t in prev_topics])
        next_vectors = np.array([self._topic_to_vector(t) for t in next_topics])

        # Compute cosine similarity matrix
        self.similarity_matrix = cosine_similarity(prev_vectors, next_vectors)
        return self.similarity_matrix

    def generate_sankey(self, threshold=0.1):
        """Generate a Sankey diagram of topic evolution."""
        labels = []
        sources = []
        targets = []
        values = []

        # Build node labels
        label_map = {}
        node_id = 0
        for period, topics in self.time_slices.items():
            for tid in topics:
                label = f"{period}-T{tid}"
                label_map[(period, tid)] = node_id
                labels.append(label)
                node_id += 1

        # Build links for similarities above threshold
        periods = list(self.time_slices.keys())
        for i, prev_tid in enumerate(self.time_slices[periods[0]].keys()):
            for j, next_tid in enumerate(self.time_slices[periods[1]].keys()):
                sim = self.similarity_matrix[i][j]
                if sim > threshold:
                    sources.append(label_map[(periods[0], prev_tid)])
                    targets.append(label_map[(periods[1], next_tid)])
                    values.append(sim * 100)  # scale for better visibility

        # Create Sankey figure
        fig = go.Figure(go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color="rgba(100, 149, 237, 0.6)"  # cornflower blue
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                hoverinfo='all'
            )
        ))

        fig.update_layout(
            title_text="Topic Evolution Analysis – Sankey Diagram",
            font=dict(size=14, family="Arial"),
            width=1600,
            height=900,
            margin=dict(t=80, l=50, r=50, b=50)
        )

        # Save interactive HTML
        fig.write_html("topic_evolution.html")
        return fig


# Example usage
if __name__ == "__main__":
    analyzer = TopicAnalyzer(time_slices)
    sim_matrix = analyzer.calculate_similarity()
    print("Similarity matrix:\n", sim_matrix)

    sankey_fig = analyzer.generate_sankey(threshold=0.1)
    print("Sankey diagram generated: topic_evolution.html")
