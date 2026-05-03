import numpy as np
import matplotlib.pyplot as plt


def plot_attention_weights(step_weights: np.ndarray, feature_weights: np.ndarray = None,
                           title: str = "TARA-DQN Attention Weights",
                           save_path: str = None):
    if feature_weights is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(7, 5))
        axes = [axes]

    fig.suptitle(title, fontsize=14)

    if step_weights.ndim == 3:
        avg_weights = step_weights.mean(axis=0)
    else:
        avg_weights = step_weights

    im = axes[0].imshow(avg_weights, aspect='auto', cmap='YlOrRd')
    axes[0].set_xlabel('Key Position')
    axes[0].set_ylabel('Query Position')
    axes[0].set_title('Step-Level Attention Weights')
    plt.colorbar(im, ax=axes[0])

    if feature_weights is not None and len(axes) > 1:
        if feature_weights.ndim > 1:
            avg_feat = feature_weights.mean(axis=0)
        else:
            avg_feat = feature_weights

        axes[1].bar(range(len(avg_feat)), avg_feat, color='steelblue', alpha=0.7)
        axes[1].set_xlabel('Feature Dimension')
        axes[1].set_ylabel('Gate Weight')
        axes[1].set_title('Feature-Level Attention Weights')
        axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
