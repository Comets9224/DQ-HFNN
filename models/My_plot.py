import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

def plot_training_curves(history, config, run_id=None, batch_timestamp=None, save_dir=None, is_kfold_avg=False, k_folds=None):
    """
    Plot training loss and accuracy curves for a single run

    Parameters:
        history: numpy array containing training history [train_loss, val_loss, train_acc, val_acc]
        config: Configuration object
        run_id: Run ID
        batch_timestamp: Batch timestamp
        save_dir: Optional, specifies save directory
        is_kfold_avg: Whether showing K-fold average results (affects title)
        k_folds: Number of K-folds (only valid when is_kfold_avg=True)

    Returns:
        fig_dir: Path to directory where images are saved
    """
    # Ensure history is a numpy array
    if not isinstance(history, np.ndarray):
        history = np.array(history)

    num_epochs = len(history)

    # Create save directory
    if save_dir is None:
        if batch_timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            if run_id is not None:
                fig_dir = f'figures/{config.dataset}/{config.model_type}/run_{run_id}_{timestamp}'
            else:
                fig_dir = f'figures/{config.dataset}/{config.model_type}/{timestamp}'
        else:
            if run_id is not None:
                fig_dir = f'figures/{config.dataset}/{config.model_type}/batch_{batch_timestamp}/run_{run_id}'
            else:
                fig_dir = f'figures/{config.dataset}/{config.model_type}/batch_{batch_timestamp}'
    else:
        fig_dir = save_dir

    os.makedirs(fig_dir, exist_ok=True)

    # Title suffix (K-fold notation)
    title_suffix = f' (Avg over {k_folds} folds)' if is_kfold_avg else ''

    # ===== Figure 1: Training + Validation Loss Curves =====
    plt.figure(figsize=(10, 6))
    plt.plot(history[:, 0], label='Train Loss', linewidth=2, color='#415882')
    plt.plot(history[:, 1], label='Val Loss', linewidth=2, color='#8BCFDC')
    plt.legend(fontsize=12)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'Training & Validation Loss - {config.dataset.upper()}{title_suffix}', fontsize=14)
    plt.xticks(np.arange(0, num_epochs + 1, step=max(1, num_epochs // 10)))
    plt.grid(True, alpha=0.3)

    loss_path = os.path.join(fig_dir, 'loss_curve.png')
    plt.savefig(loss_path, dpi=300, bbox_inches='tight')
    plt.close()

    # ===== Figure 2: Training + Validation Accuracy Curves =====
    plt.figure(figsize=(10, 6))
    plt.plot(history[:, 2], label='Train Accuracy', linewidth=2, color='#C5C1D8')
    plt.plot(history[:, 3], label='Val Accuracy', linewidth=2, color='#67A6C2')
    plt.legend(fontsize=12)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'Training & Validation Accuracy - {config.dataset.upper()}{title_suffix}', fontsize=14)
    plt.xticks(np.arange(0, num_epochs + 1, step=max(1, num_epochs // 10)))
    plt.yticks(np.arange(0, 1.05, 0.1))
    plt.grid(True, alpha=0.3)

    acc_path = os.path.join(fig_dir, 'accuracy_curve.png')
    plt.savefig(acc_path, dpi=300, bbox_inches='tight')
    plt.close()

    # ===== Figure 3: Three-in-One Plot =====
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Cross Entropy Loss/Accuracy', fontsize=12)

    line1 = ax1.plot(history[:, 0], label='Training loss', linewidth=2, color='#415882')
    line2 = ax1.plot(history[:, 2], label='Training accuracy', linewidth=2, color='#C5C1D8')
    line3 = ax1.plot(history[:, 3], label='Validation accuracy', linewidth=2, color='#67A6C2')

    ax1.set_ylim(0.0, 1.0)
    ax1.tick_params(axis='y')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(np.arange(0, num_epochs + 1, step=max(1, num_epochs // 10)))

    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='right', fontsize=11)

    plt.title(f'Convergence Analysis - {config.dataset.upper()}{title_suffix}', fontsize=14)
    plt.tight_layout()

    combined_path = os.path.join(fig_dir, 'combined_curve.png')
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Save history data
    history_path = os.path.join(fig_dir, 'history.npy')
    np.save(history_path, history)

    return fig_dir


def plot_comparison_curves(histories_dict, save_dir='figures/comparison'):
    """
    Compare training curves across multiple models

    Parameters:
        histories_dict: Dictionary in format {'model_name': history_array, ...}
        save_dir: Save directory
    """
    os.makedirs(save_dir, exist_ok=True)

    # Plot loss comparison
    plt.figure(figsize=(12, 8))
    for model_name, history in histories_dict.items():
        if isinstance(history, list):
            history = np.array(history)
        plt.plot(history[:, 1], label=f'{model_name} Val Loss', linewidth=2)

    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.title('Model Comparison - Validation Loss')
    plt.grid(True, alpha=0.3)

    loss_comp_path = os.path.join(save_dir, 'loss_comparison.png')
    plt.savefig(loss_comp_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Loss comparison saved to: {loss_comp_path}")

    # Plot accuracy comparison
    plt.figure(figsize=(12, 8))
    for model_name, history in histories_dict.items():
        if isinstance(history, list):
            history = np.array(history)
        plt.plot(history[:, 3], label=f'{model_name} Val Acc', linewidth=2)

    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.title('Model Comparison - Validation Accuracy')
    plt.grid(True, alpha=0.3)

    acc_comp_path = os.path.join(save_dir, 'accuracy_comparison.png')
    plt.savefig(acc_comp_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Accuracy comparison saved to: {acc_comp_path}")

    return save_dir


def plot_metrics_summary(best_acc_list, best_prec_list, best_rec_list, best_f1_list,
                         config, save_dir=None):
    """
    Plot aggregated metrics across multiple runs

    Parameters:
        best_acc_list, best_prec_list, best_rec_list, best_f1_list: Lists of each metric
        config: Configuration object
        save_dir: Save directory
    """
    if save_dir is None:
        save_dir = f'figures/{config.dataset}/{config.model_type}'

    os.makedirs(save_dir, exist_ok=True)

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [best_acc_list, best_prec_list, best_rec_list, best_f1_list]

    plt.figure(figsize=(12, 6))

    # Create box plots
    plt.boxplot(values, labels=metrics)
    plt.ylabel('Score (%)')
    plt.title(f'Performance Summary - {config.dataset.upper()} ({config.model_type})')
    plt.grid(True, alpha=0.3)

    # Add numerical labels
    for i, metric_values in enumerate(values):
        mean_val = np.mean(metric_values)
        plt.text(i + 1, mean_val, f'{mean_val:.2f}%',
                 ha='center', va='bottom', fontweight='bold')

    summary_path = os.path.join(save_dir, 'metrics_summary.png')
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Metrics summary saved to: {summary_path}")

    return save_dir


def plot_multi_run_summary(all_histories, config, batch_timestamp, save_dir=None):
    """
    Plot aggregated charts across multiple runs (mean + variance)

    Parameters:
        all_histories: list of numpy arrays, each element being a complete run's history
        config: Configuration object
        batch_timestamp: Batch timestamp (used for saving to corresponding batch directory)
        save_dir: Save directory (optional, overrides batch_timestamp if specified)

    Returns:
        summary_dir: Directory path where summary results are saved
    """
    if len(all_histories) == 0:
        print("Warning: No complete runs found. Skipping summary plots.")
        return None

    # Create summary directory
    if save_dir is None:
        summary_dir = f'figures/{config.dataset}/{config.model_type}/batch_{batch_timestamp}/summary'
    else:
        summary_dir = save_dir

    os.makedirs(summary_dir, exist_ok=True)

    # Convert to numpy array (num_runs, num_epochs, 4)
    all_histories = np.array(all_histories)
    num_runs, num_epochs, _ = all_histories.shape

    # print(f"\n{'=' * 60}")
    # print(f"Generating summary for {num_runs} complete runs")
    # print(f"Batch timestamp: {batch_timestamp}")
    # print(f"{'=' * 60}")

    # Calculate mean and standard deviation
    mean_history = np.mean(all_histories, axis=0)
    std_history = np.std(all_histories, axis=0)

    epochs = np.arange(num_epochs)

    # ===== Figure 1: Mean Training + Validation Loss =====
    plt.figure(figsize=(10, 6))
    plt.plot(mean_history[:, 0], label='Mean Train Loss', linewidth=2, color='#415882')
    plt.fill_between(epochs,
                     mean_history[:, 0] - std_history[:, 0],
                     mean_history[:, 0] + std_history[:, 0],
                     alpha=0.2, color='#415882')

    plt.plot(mean_history[:, 1], label='Mean Val Loss', linewidth=2, color='#8BCFDC')
    plt.fill_between(epochs,
                     mean_history[:, 1] - std_history[:, 1],
                     mean_history[:, 1] + std_history[:, 1],
                     alpha=0.2, color='#8BCFDC')

    plt.legend(fontsize=12)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'Mean Loss over {num_runs} Runs - {config.dataset.upper()}', fontsize=14)
    plt.xticks(np.arange(0, num_epochs + 1, step=max(1, num_epochs // 10)))
    plt.grid(True, alpha=0.3)

    mean_loss_path = os.path.join(summary_dir, 'mean_loss_curve.png')
    plt.savefig(mean_loss_path, dpi=300, bbox_inches='tight')
    plt.close()
    #print(f"Mean loss curve saved to: {mean_loss_path}")

    # ===== Figure 2: Mean Training + Validation Accuracy =====
    plt.figure(figsize=(10, 6))
    plt.plot(mean_history[:, 2], label='Mean Train Accuracy', linewidth=2, color='#C5C1D8')
    plt.fill_between(epochs,
                     mean_history[:, 2] - std_history[:, 2],
                     mean_history[:, 2] + std_history[:, 2],
                     alpha=0.2, color='#C5C1D8')

    plt.plot(mean_history[:, 3], label='Mean Val Accuracy', linewidth=2, color='#67A6C2')
    plt.fill_between(epochs,
                     mean_history[:, 3] - std_history[:, 3],
                     mean_history[:, 3] + std_history[:, 3],
                     alpha=0.2, color='#67A6C2')

    plt.legend(fontsize=12)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'Mean Accuracy over {num_runs} Runs - {config.dataset.upper()}', fontsize=14)
    plt.xticks(np.arange(0, num_epochs + 1, step=max(1, num_epochs // 10)))
    plt.yticks(np.arange(0, 1.05, 0.1))
    plt.grid(True, alpha=0.3)

    mean_acc_path = os.path.join(summary_dir, 'mean_accuracy_curve.png')
    plt.savefig(mean_acc_path, dpi=300, bbox_inches='tight')
    plt.close()
    #print(f"Mean accuracy curve saved to: {mean_acc_path}")

    # ===== Figure 3: Mean Three-in-One Plot =====
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Cross Entropy Loss/Accuracy', fontsize=12)

    line1 = ax1.plot(mean_history[:, 0], label='Mean Training loss',
                     linewidth=2, color='#415882')
    ax1.fill_between(epochs,
                     mean_history[:, 0] - std_history[:, 0],
                     mean_history[:, 0] + std_history[:, 0],
                     alpha=0.15, color='#415882')

    line2 = ax1.plot(mean_history[:, 2], label='Mean Training accuracy',
                     linewidth=2, color='#C5C1D8')
    ax1.fill_between(epochs,
                     mean_history[:, 2] - std_history[:, 2],
                     mean_history[:, 2] + std_history[:, 2],
                     alpha=0.15, color='#C5C1D8')

    line3 = ax1.plot(mean_history[:, 3], label='Mean Validation accuracy',
                     linewidth=2, color='#67A6C2')
    ax1.fill_between(epochs,
                     mean_history[:, 3] - std_history[:, 3],
                     mean_history[:, 3] + std_history[:, 3],
                     alpha=0.15, color='#67A6C2')

    ax1.set_ylim(0.0, 1.0)
    ax1.tick_params(axis='y')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(np.arange(0, num_epochs + 1, step=max(1, num_epochs // 10)))

    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='right', fontsize=11)

    plt.title(f'Mean Convergence over {num_runs} Runs - {config.dataset.upper()}', fontsize=14)
    plt.tight_layout()

    mean_combined_path = os.path.join(summary_dir, 'mean_combined_curve.png')
    plt.savefig(mean_combined_path, dpi=300, bbox_inches='tight')
    plt.close()
    #print(f"Mean combined curve saved to: {mean_combined_path}")

    # ===== Figure 4: Final Metrics Box Plot =====
    best_val_accs = [hist[:, 3].max() for hist in all_histories]
    final_train_accs = [hist[-1, 2] for hist in all_histories]
    final_val_accs = [hist[-1, 3] for hist in all_histories]
    final_train_losses = [hist[-1, 0] for hist in all_histories]
    final_val_losses = [hist[-1, 1] for hist in all_histories]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Subplot 1: Accuracy Box Plot
    ax1 = axes[0]
    acc_data = [final_train_accs, final_val_accs, best_val_accs]
    bp1 = ax1.boxplot(acc_data, labels=['Final Train Acc', 'Final Val Acc', 'Best Val Acc'],
                      patch_artist=True, widths=0.6)

    colors = ['#C5C1D8', '#67A6C2', '#8BCFDC']
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title(f'Accuracy Distribution ({num_runs} runs)', fontsize=13)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 1.05)

    for i, data in enumerate(acc_data):
        mean_val = np.mean(data)
        std_val = np.std(data)
        ax1.text(i + 1, mean_val, f'{mean_val:.3f}\n±{std_val:.3f}',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Subplot 2: Loss Box Plot
    ax2 = axes[1]
    loss_data = [final_train_losses, final_val_losses]
    bp2 = ax2.boxplot(loss_data, labels=['Final Train Loss', 'Final Val Loss'],
                      patch_artist=True, widths=0.6)

    colors = ['#415882', '#8BCFDC']
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title(f'Loss Distribution ({num_runs} runs)', fontsize=13)
    ax2.grid(True, alpha=0.3, axis='y')

    for i, data in enumerate(loss_data):
        mean_val = np.mean(data)
        std_val = np.std(data)
        ax2.text(i + 1, mean_val, f'{mean_val:.3f}\n±{std_val:.3f}',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    boxplot_path = os.path.join(summary_dir, 'metrics_boxplot.png')
    plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
    plt.close()
    #print(f"Metrics boxplot saved to: {boxplot_path}")

    # ===== Save Data =====
    mean_history_path = os.path.join(summary_dir, 'mean_history.npy')
    np.save(mean_history_path, mean_history)
    #print(f"Mean history saved to: {mean_history_path}")

    all_runs_path = os.path.join(summary_dir, 'all_runs_history.npy')
    np.save(all_runs_path, all_histories)
    #print(f"All runs history saved to: {all_runs_path}")

    # Save statistical summary
    summary_stats = {
        'batch_timestamp': batch_timestamp,
        'num_runs': num_runs,
        'num_epochs': num_epochs,
        'best_val_acc_mean': np.mean(best_val_accs),
        'best_val_acc_std': np.std(best_val_accs),
        'final_train_acc_mean': np.mean(final_train_accs),
        'final_train_acc_std': np.std(final_train_accs),
        'final_val_acc_mean': np.mean(final_val_accs),
        'final_val_acc_std': np.std(final_val_accs),
        'final_train_loss_mean': np.mean(final_train_losses),
        'final_train_loss_std': np.std(final_train_losses),
        'final_val_loss_mean': np.mean(final_val_losses),
        'final_val_loss_std': np.std(final_val_losses),
    }

    summary_txt_path = os.path.join(summary_dir, 'summary_stats.txt')
    with open(summary_txt_path, 'w') as f:
        f.write(f"Summary Statistics for Batch {batch_timestamp}\n")
        f.write(f"{'=' * 60}\n\n")
        for key, value in summary_stats.items():
            if isinstance(value, str):
                f.write(f"{key}: {value}\n")
            else:
                f.write(f"{key}: {value:.6f}\n")

    #print(f"Summary statistics saved to: {summary_txt_path}")
    #print(f"\n{'=' * 60}")
    #print(f"Summary complete! All files saved to: {summary_dir}")
    #print(f"{'=' * 60}\n")

    return summary_dir


def collect_batch_histories(base_dir, batch_timestamp, expected_epochs):
    """
    Collect history data from all complete runs in specified batch

    Parameters:
        base_dir: Base directory (e.g., 'figures/dmnist/joint_membership')
        batch_timestamp: Batch timestamp
        expected_epochs: Expected number of epochs

    Returns:
        all_histories: list of numpy arrays
        valid_run_ids: List of valid run IDs
    """
    all_histories = []
    valid_run_ids = []

    batch_dir = os.path.join(base_dir, f'batch_{batch_timestamp}')

    if not os.path.exists(batch_dir):
        print(f"Warning: Batch directory {batch_dir} does not exist.")
        return all_histories, valid_run_ids

    # Find all run_* directories (excluding summary)
    run_dirs = [d for d in os.listdir(batch_dir)
                if d.startswith('run_') and d != 'summary']

    print(f"\n{'=' * 60}")
    print(f"Collecting histories from batch: {batch_timestamp}")
    print(f"Batch directory: {batch_dir}")
    print(f"Expected epochs: {expected_epochs}")
    print(f"Found {len(run_dirs)} run directories")
    print(f"{'=' * 60}")

    for run_dir in sorted(run_dirs):
        history_path = os.path.join(batch_dir, run_dir, 'history.npy')

        if os.path.exists(history_path):
            try:
                history = np.load(history_path)

                if len(history) == expected_epochs:
                    all_histories.append(history)
                    run_id = run_dir.replace('run_', '')
                    valid_run_ids.append(run_id)
                    print(f"  ✓ {run_dir}: Complete ({len(history)} epochs)")
                else:
                    print(f"  ✗ {run_dir}: Incomplete ({len(history)}/{expected_epochs} epochs) - SKIPPED")
            except Exception as e:
                print(f"  ✗ {run_dir}: Error loading - {e}")
        else:
            print(f"  ✗ {run_dir}: No history.npy found")

    print(f"\nTotal complete runs in this batch: {len(all_histories)}")
    print(f"{'=' * 60}\n")

    return all_histories, valid_run_ids