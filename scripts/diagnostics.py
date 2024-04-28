import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json
if __name__ == "__main__":

    with open("data/results_base.json") as f:
        base_data = json.load(f)

    with open("data/results_meditron.json") as f:
        meditron_data = json.load(f)

    with open("data/results_biomistral.json") as f:
        biomistral_data = json.load(f)

    data = pd.DataFrame.from_dict(
        {
            "case": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Mean"],
            "Mistral-7B": list(base_data.values()),
            "Meditron-7B": list(meditron_data.values()),
            "BioMistral-7B": list(biomistral_data.values())
        }
    )

    data_reshaped = data.melt(id_vars='case', var_name='method', value_name='value')

    data.set_index("case", inplace=True)
    plt.figure(figsize=(12, 6))
    sns.barplot(x="case", y="value", hue="method", data=data_reshaped, palette="colorblind")

    plt.legend(title="Architecture")
    plt.xticks(rotation=45, fontstyle="italic")
    plt.xlabel("Case")
    plt.ylabel("Average Weighted Similarity")
    plt.title("Average Weighted Similarity across 5 Trials for Each Case")

    plt.savefig("output/diagnostics.png", bbox_inches="tight", dpi=300)
