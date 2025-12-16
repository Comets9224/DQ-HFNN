"""
JAFFE Dataset-Specific Training Script (Dual-Qubit Joint Membership Model + K-Fold Cross Validation)

Features:
- Exclusively supports JAFFE dataset
- Uses dual-qubit joint membership model (My_joint_jaffe.py)
- Supports K-Fold cross validation mode
- Includes complete training, validation, saving and visualization pipeline
"""

import torch
import os
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import datetime
import time
import logging
import sys
from argparse import Namespace

from models.My_joint_jaffe import JAFFE_JointMembership
from data.jaffe.data_loader import load_dataset_jaffe
from utils.evaluation import cal_ms, cal_merics
from models.My_plot import plot_training_curves, plot_multi_run_summary

print("✓ Loaded: JAFFE joint membership model from My_joint_jaffe.py")


def init_logger(filename, logger_name):
    """初始化日志系统"""
    timestamp = datetime.datetime.utcnow().strftime('%Y%m%d_%H-%M-%S')

    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(filename=filename),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(logger_name)
    logger.info('### Init. Logger {} ###'.format(logger_name))
    return logger


my_logger = init_logger("./run_result_jaffe.log", "jaffe_logger")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def trainModel(config, run_id=None, batch_timestamp=None):
    """
    Train JAFFE model (supports K-Fold)

    Parameters:
        config: Configuration object
        run_id: Run ID (for multiple runs)
        batch_timestamp: Batch timestamp (for save paths)

    Returns:
        model: Trained model
        history: Training history [epochs, 4] (train_loss, val_loss, train_acc, val_acc)
        best_acc_list: List of best accuracy scores
        best_prec_list: List of best precision scores
        best_rec_list: List of best recall scores
        best_f1_list: List of best F1 scores
    """
    global best_model_state

    if batch_timestamp and run_id:
        if hasattr(config, 'fold'):
            checkpoints_path = os.path.join(
                config.ckpt_path,
                'jaffe',
                'joint_membership',
                f'batch_{batch_timestamp}',
                f'run_{run_id}'
            )
        else:
            checkpoints_path = os.path.join(
                config.ckpt_path,
                'jaffe',
                'joint_membership',
                f'batch_{batch_timestamp}',
                f'run_{run_id}'
            )
    else:

        checkpoints_path = os.path.join(config.ckpt_path, 'jaffe', 'joint_membership')

    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path, exist_ok=True)
        print(f"{checkpoints_path} is created")

    batchsize = config.batch_size
    epochs = config.epochs
    LR = config.lr

    k = config.k
    fold = config.fold
    data_train, data_test = load_dataset_jaffe(
        k=k,
        fold=fold,
        image_size=32
    )

    num_classes = 7
    my_logger.info(f"Using dataset: JAFFE, model: joint_membership, num_classes: {num_classes}")
    my_logger.info(f"K-fold mode: k={k}, fold={fold}")

    train_data_size = len(data_train)
    valid_data_size = len(data_test)
    print(f'JAFFE Dataset, train_size: {train_data_size:4d}, valid_size: {valid_data_size:4d}')

    train_loader = DataLoader(
        data_train,
        batch_size=batchsize,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        data_test,
        batch_size=batchsize,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    hidden_dim = getattr(config, 'hidden_dim', 512)
    n_qnn_layers = getattr(config, 'n_qnn_layers', 3)
    n_random_pairs = getattr(config, 'n_random_pairs', 154)

    model = JAFFE_JointMembership(
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        n_qnn_layers=n_qnn_layers,
        n_random_pairs=n_random_pairs
    ).to(device)

    my_logger.info(f"Using model: {model.__class__.__name__}")
    my_logger.info(f'The model has {count_parameters(model):,} trainable parameters')

    criterion = nn.CrossEntropyLoss()

    optim_type = getattr(config, 'optim_type', 'AdamW')
    if optim_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-3)
        my_logger.info(f"Using Adam optimizer with lr={LR}")
    elif optim_type == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
        my_logger.info(f"Using AdamW optimizer with lr={LR}, weight_decay=1e-2")
    elif optim_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-3)
        my_logger.info(f"Using SGD optimizer with lr={LR}, momentum=0.9")
    else:
        raise ValueError(f"Unsupported optimizer type: {optim_type}")

    my_logger.info(f"Initial learning rate: {LR}")

    warmup_epochs = getattr(config, 'warmup_epochs', 3)

    if warmup_epochs > 0:
        # Warm-up + CosineAnnealingLR
        def warmup_cosine_lambda(epoch):
            if epoch < warmup_epochs:

                return (epoch + 1) / warmup_epochs
            else:

                progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
                min_lr_ratio = 1e-5 / LR  # eta_min 与初始学习率的比例
                return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + np.cos(np.pi * progress))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_lambda)
        my_logger.info(f"Using Warm-up ({warmup_epochs} epochs) + CosineAnnealingLR (eta_min=1e-5)")
    else:

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=epochs,
            eta_min=1e-5
        )
        my_logger.info(f"Using CosineAnnealingLR: T_max={epochs}, eta_min=1e-5")
    history = []
    best_acc = 0
    best_epoch = 0
    best_model_state = None

    precision_t = []
    recall_t = []
    f1_t = []

    for epoch in range(epochs):
        epoch_start = time.time()
        my_logger.info("Epoch: {}/{}".format(epoch + 1, epochs))
        model.train()
        train_loss = 0.0
        train_acc = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

            _, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            train_acc += acc.item() * inputs.size(0)

        model.eval()
        valid_loss = 0.0
        valid_acc = 0.0
        y_true = []
        y_pred = []

        with torch.no_grad():
            for j, (inputs, labels) in enumerate(test_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item() * inputs.size(0)

                _, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))
                acc = torch.mean(correct_counts.type(torch.FloatTensor))
                valid_acc += acc.item() * inputs.size(0)

                y_true.extend(labels.cpu().numpy().tolist())
                y_pred.extend(predictions.cpu().numpy().tolist())

        scheduler.step()
        my_logger.info('\t last_lr: ' + str(scheduler.get_last_lr()[0]))

        avg_train_loss = train_loss / train_data_size
        avg_train_acc = train_acc / train_data_size
        avg_valid_loss = valid_loss / valid_data_size
        avg_valid_acc = valid_acc / valid_data_size

        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])
        if best_acc < avg_valid_acc:
            best_acc = avg_valid_acc
            best_epoch = epoch + 1
            best_model_state = model.state_dict()

        epoch_end = time.time()

        precision, recall, f1 = cal_merics(y_true, y_pred)
        precision_t.append(precision)
        recall_t.append(recall)
        f1_t.append(f1)

        my_logger.info(
            "\t Training: Loss: {:.4f}, Accuracy: {:.4f}%".format(
                avg_train_loss, avg_train_acc * 100))
        my_logger.info(
            "\t Validation: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.3f}s".format(
                avg_valid_loss, avg_valid_acc * 100, epoch_end - epoch_start))
        my_logger.info("\t Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(
            precision, recall, f1))
        my_logger.info("\t Best Accuracy for validation: {:.4f} at epoch {:03d}".format(
            best_acc, best_epoch))

    best_prec = precision_t[best_epoch - 1]
    best_rec = recall_t[best_epoch - 1]
    best_f1 = f1_t[best_epoch - 1]

    best_acc_list = [best_acc * 100]
    best_prec_list = [best_prec * 100]
    best_rec_list = [best_rec * 100]
    best_f1_list = [best_f1 * 100]

    try:
        model.load_state_dict(best_model_state)
        model_filename = f'best_fold{config.fold}.pt'
        model_save_path = os.path.join(checkpoints_path, model_filename)
        torch.save(model, model_save_path)
        my_logger.info(f"\n✓ Final best model saved: {model_filename} (Acc={best_acc:.4f}, Epoch={best_epoch})")
    except Exception as e:
        my_logger.error(f"\n✗ Failed to save final best model: {e}")

    history = np.array(history)
    return model, history, best_acc_list, best_prec_list, best_rec_list, best_f1_list


def PredictModel(config):
    batchsize = config.batch_size
    run_num = getattr(config, 'predict_run', 1)
    fold_num = getattr(config, 'predict_fold', 0)
    batch_timestamp = getattr(config, 'batch_timestamp', None)

    my_logger.info(
        f"PredictModel using dataset: JAFFE, model: joint_membership, "
        f"run: {run_num}, fold: {fold_num}")

    k = config.k
    data_train, data_test = load_dataset_jaffe(
        k=k,
        fold=fold_num,
        image_size=32
    )

    valid_data_size = len(data_test)
    test_loader = DataLoader(data_test, batch_size=batchsize, shuffle=False)

    if batch_timestamp:
        base_path = os.path.join('./checkpoints', 'jaffe', 'joint_membership',
                                 f'batch_{batch_timestamp}', f'run_{run_num}')
    else:
        base_path = os.path.join('./checkpoints', 'jaffe', 'joint_membership', f'run_{run_num}')

    model_path = os.path.join(base_path, f'best_fold{fold_num}.pt')

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Please check:\n"
            f"  - run_num={run_num}, fold_num={fold_num}\n"
            f"  - batch_timestamp={batch_timestamp}"
        )

    my_logger.info(f"Loading model from: {model_path}")
    model = torch.load(model_path, map_location=device)
    model.eval()

    valid_acc = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for j, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            valid_acc += acc.item() * inputs.size(0)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(predictions.cpu().numpy().tolist())

    avg_valid_acc = valid_acc / valid_data_size
    print("Accuracy: {:.4f}%".format(avg_valid_acc * 100))

    return y_true, y_pred, model.parameters()


if __name__ == '__main__':

    train_flag = True

    config = Namespace(

        project_name='jaffe_2qubit_kfold',

        batch_size=8,
        lr=5e-4,
        optim_type='AdamW',
        warmup_epochs=3,
        epochs=100,

        dataset='jaffe',
        model_type='joint_membership',

        hidden_dim=128,
        n_qnn_layers=3,
        n_random_pairs=154,  # Number of random pairs (30%), can be adjusted to 256(50%) or 512(100%)

        k=8,
        num_runs=5,

        ckpt_path='./checkpoints',

        predict_run=1,
        predict_fold=0,
        batch_timestamp=None,
    )

    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    my_logger.info(f"Experiment started at: {nowtime}")
    my_logger.info(f"Configuration: {config}")

    if train_flag:

        all_start = time.time()
        batch_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        k_folds = config.k
        num_runs = config.num_runs

        my_logger.info(f"✅ K-Fold Cross-Validation Enabled: {k_folds} folds, {num_runs} runs")

        all_fold_best_acc = []
        all_fold_best_prec = []
        all_fold_best_rec = []
        all_fold_best_f1 = []

        for run_id in range(1, num_runs + 1):
            my_logger.info(f"\n{'=' * 70}")
            my_logger.info(f"║  RUN {run_id}/{num_runs}")
            my_logger.info(f"{'=' * 70}\n")

            run_start = time.time()
            run_fold_histories = []
            run_fold_best_acc = []
            run_fold_best_prec = []
            run_fold_best_rec = []
            run_fold_best_f1 = []

            for fold_idx in range(k_folds):
                config.fold = fold_idx
                my_logger.info(f"\n--- Fold {fold_idx + 1}/{k_folds} ---")

                model, fold_history, best_acc_list, best_prec_list, best_rec_list, best_f1_list = trainModel(
                    config,
                    run_id=run_id,
                    batch_timestamp=batch_timestamp
                )

                run_fold_best_acc.extend(best_acc_list)
                run_fold_best_prec.extend(best_prec_list)
                run_fold_best_rec.extend(best_rec_list)
                run_fold_best_f1.extend(best_f1_list)

                if len(best_acc_list) > 0:
                    my_logger.info(f"  ✓ Fold {fold_idx + 1} completed: Best Acc={best_acc_list[0]:.2f}%")

                if len(fold_history) == config.epochs:
                    run_fold_histories.append(fold_history)
                else:
                    my_logger.warning(
                        f"  ⚠️  Fold {fold_idx + 1} incomplete: {len(fold_history)}/{config.epochs} epochs")

            run_end = time.time()

            if len(run_fold_best_acc) > 0:
                run_mean_acc = np.mean(run_fold_best_acc)
                run_mean_prec = np.mean(run_fold_best_prec)
                run_mean_rec = np.mean(run_fold_best_rec)
                run_mean_f1 = np.mean(run_fold_best_f1)

                all_fold_best_acc.extend(run_fold_best_acc)
                all_fold_best_prec.extend(run_fold_best_prec)
                all_fold_best_rec.extend(run_fold_best_rec)
                all_fold_best_f1.extend(run_fold_best_f1)

                my_logger.info(f"\n{'=' * 70}")
                my_logger.info(f"║  Run {run_id} Summary (Averaged over {k_folds} folds)")
                my_logger.info(f"{'=' * 70}")
                my_logger.info(f"Mean Accuracy:  {run_mean_acc:.4f}% ± {np.std(run_fold_best_acc):.4f}%")
                my_logger.info(f"Mean Precision: {run_mean_prec:.4f}% ± {np.std(run_fold_best_prec):.4f}%")
                my_logger.info(f"Mean Recall:    {run_mean_rec:.4f}% ± {np.std(run_fold_best_rec):.4f}%")
                my_logger.info(f"Mean F1:        {run_mean_f1:.4f}% ± {np.std(run_fold_best_f1):.4f}%")
                my_logger.info(f"Time: {run_end - run_start:.2f}s")
                my_logger.info(f"{'=' * 70}\n")

            if len(run_fold_histories) > 0:
                fold_lengths = [len(h) for h in run_fold_histories]
                min_length = min(fold_lengths)
                max_length = max(fold_lengths)

                if min_length != max_length:
                    my_logger.warning(
                        f"⚠️  K-fold histories have different lengths: min={min_length}, max={max_length}")
                    my_logger.warning(f"   Truncating all histories to {min_length} epochs for averaging")
                    run_fold_histories = [h[:min_length] for h in run_fold_histories]

                try:
                    run_mean_history = np.mean(run_fold_histories, axis=0)

                    plot_training_curves(
                        run_mean_history,
                        config,
                        run_id=run_id,
                        batch_timestamp=batch_timestamp,
                        is_kfold_avg=True,
                        k_folds=k_folds
                    )
                    my_logger.info(f"✅ K-fold averaged plot saved (based on {min_length} epochs)")
                except Exception as e:
                    my_logger.error(f"❌ Failed to generate K-fold average plot: {e}")

        my_logger.info("\n" + "=" * 70)
        my_logger.info("║  FINAL RESULTS (Based on All Individual Folds)")
        my_logger.info("=" * 70)
        my_logger.info(
            f"Total independent tests: {len(all_fold_best_acc)} folds ({num_runs} runs × {k_folds} folds)")
        my_logger.info(f"")
        my_logger.info(
            f"Overall Accuracy:  {np.mean(all_fold_best_acc):.4f}% ± {np.std(all_fold_best_acc):.4f}%")
        my_logger.info(
            f"Overall Precision: {np.mean(all_fold_best_prec):.4f}% ± {np.std(all_fold_best_prec):.4f}%")
        my_logger.info(
            f"Overall Recall:    {np.mean(all_fold_best_rec):.4f}% ± {np.std(all_fold_best_rec):.4f}%")
        my_logger.info(
            f"Overall F1:        {np.mean(all_fold_best_f1):.4f}% ± {np.std(all_fold_best_f1):.4f}%")
        my_logger.info(f"")
        my_logger.info(f"Range: [{np.min(all_fold_best_acc):.2f}%, {np.max(all_fold_best_acc):.2f}%]")
        my_logger.info("=" * 70)

        all_end = time.time()
        all_time = round(all_end - all_start)
        print(f'Total time: {all_time} seconds')
        print("Total Time: {:d} min {:d} sec".format(all_time // 60, all_time % 60))

    else:
        y_true, y_pred, qnn_params = PredictModel(config)
        print(f"Number of test samples: {len(y_true)}")
        print(f"Prediction completed")
        # Display first few prediction results
        print("First 5 prediction examples:")
        for i in range(min(5, len(y_true))):
            status = "Correct" if y_true[i] == y_pred[i] else "Incorrect"
            print(f"  Sample {i + 1}: True={y_true[i]}, Predicted={y_pred[i]} ({status})")
