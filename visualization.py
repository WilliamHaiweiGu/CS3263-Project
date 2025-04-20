import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    models = ["Naive Bayesian Classifier", "TFIDF + ANN", "XLNet"]
    macro_f1 = [0.710, 0.587, 0.817]
    accuracy = [0.830, 0.901, 0.902]

    x = np.arange(len(models))
    width = 1 / len(models)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, macro_f1, width, label='Macro F1', color='#2b83ba')
    bars2 = ax.bar(x + width / 2, accuracy, width, label='Accuracy', color='#fdae61')

    ax.set_title('Performance by Model')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.set_ylim([0, 1])
    ax.set_ylabel('Score')
    ax.legend()
    ax.yaxis.grid(True, linestyle=':', alpha=0.5)
    # Add value labels on top of bars
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f"{height:.3f}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.subplots_adjust(left=0.1)
    plt.show()
