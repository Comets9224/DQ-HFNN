"""
Fashion MNIST Dedicated Training Script (Lightweight Version)
- Exclusively supports Fashion MNIST dataset
- Only supports dual-qubit joint membership model
- Removed all other dataset and model code
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

from data.FashionMnist.data_loader import load_dataset_fashionmnist
from models.My_joint_dmnist import DirtyMnist_JointMembership
from utils.evaluation import cal_ms, cal_merics
from models.My_plot import plot_training_curves, plot_multi_run_summary

print("✓ Loaded: Fashion MNIST joint membership model from My_joint_dmnist.py")


def load_external_params():
    import json
    params_json = os.environ.get('FASHIONMNIST_TRAIN_PARAMS')
    if params_json:
        try:
            params = json.loads(params_json)
            print(f"✅ Loaded external parameters: {params}")
            return params
        except Exception as e:
            print(f"⚠️  Failed to parse external parameters: {e}")
    return {}


EXTERNAL_PARAMS = load_external_params()


def init_logger(filename, logger_name):
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


my_logger = init_logger("./run_result_fashionmnist.log", "fashionmnist_logger")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def trainModel(config, run_id=None, batch_timestamp=None):
    """
    Fashion MNIST Dedicated Training Function

    Parameters:
        config: Configuration object (Namespace)
        run_id: Run ID (for multiple runs)
        batch_timestamp: Batch timestamp (for organizing checkpoint directories)
    """
    if batch_timestamp and run_id:
        checkpoints_path = os.path.join(
            config.ckpt_path,
            'fashionmnist',
            'joint_membership',
            f'batch_{batch_timestamp}',
            f'run_{run_id}'
        )
    else:
        checkpoints_path = os.path.join(config.ckpt_path, 'fashionmnist', 'joint_membership')

    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path, exist_ok=True)
        print(f"{checkpoints_path} is created")

    batchsize = config.batch_size
    epochs = config.epochs
    LR = config.lr

    data_train, data_test = load_dataset_fashionmnist()
    num_classes = 10

    train_data_size = len(data_train)
    valid_data_size = len(data_test)
    my_logger.info(f'Dataset: Fashion MNIST, train_size: {train_data_size:4d}, valid_size: {valid_data_size:4d}')

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

    fusion_dim = getattr(config, 'fusion_dim', 128)
    n_random_pairs = getattr(config, 'n_random_pairs', 117)  # Default 30% random pairing

    model = DirtyMnist_JointMembership(
        num_classes=num_classes,
        fusion_dim=fusion_dim,
        n_random_pairs=n_random_pairs
    ).to(device)

    my_logger.info(f"Using model: {model.__class__.__name__}")
    my_logger.info(f'The model has {count_parameters(model):,} trainable parameters')

    criterion = nn.CrossEntropyLoss()

    optim_type = getattr(config, 'optim_type', 'SGD')
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

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        milestones=[int(epochs * 0.56), int(epochs * 0.78)],
        gamma=0.1,
        last_epoch=-1
    )
    my_logger.info(f"Using MultiStepLR: milestones=[{int(epochs * 0.56)}, {int(epochs * 0.78)}], gamma=0.1")

    history = []
    best_acc_list = []
    best_prec_list = []
    best_rec_list = []
    best_f1_list = []

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

    best_prec_list.append(best_prec * 100)
    best_rec_list.append(best_rec * 100)
    best_f1_list.append(best_f1 * 100)
    best_acc_list.append(best_acc * 100)

    try:
        model.load_state_dict(best_model_state)
        model_filename = f'best_epoch{best_epoch:02d}.pt'
        model_save_path = os.path.join(checkpoints_path, model_filename)
        torch.save(model, model_save_path)
        my_logger.info(f"\n✓ Final best model saved: {model_filename} (Acc={best_acc:.4f}, Epoch={best_epoch})")
    except Exception as e:
        my_logger.error(f"\n✗ Failed to save final best model: {e}")

    history = np.array(history)
    return model, history, best_acc_list, best_prec_list, best_rec_list, best_f1_list


def PredictModel(config):
    """
    Fashion MNIST Dedicated Prediction Function

    Parameters:
        config: Configuration object (must include predict_run, predict_epoch, batch_timestamp)
    """
    batchsize = config.batch_size

    run_num = getattr(config, 'predict_run', 1)
    epoch_num = getattr(config, 'predict_epoch', 100)

    _, data_test = load_dataset_fashionmnist()
    my_logger.info(f"PredictModel using dataset: Fashion MNIST, run: {run_num}, epoch: {epoch_num}")
    valid_data_size = len(data_test)
    test_loader = DataLoader(data_test, batch_size=batchsize, shuffle=False)

    batch_timestamp = getattr(config, 'batch_timestamp', None)

    if batch_timestamp:
        base_path = os.path.join('./checkpoints', 'fashionmnist', 'joint_membership',
                                 f'batch_{batch_timestamp}', f'run_{run_num}')
    else:
        base_path = os.path.join('./checkpoints', 'fashionmnist', 'joint_membership', f'run_{run_num}')

    model_path = os.path.join(base_path, f'best_epoch{epoch_num:02d}.pt')

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Please check:\n"
            f"  - run_num={run_num}, epoch_num={epoch_num}\n"
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

    from argparse import Namespace

    base_config = {
        'project_name': 'fashionmnist_joint_membership',
        'batch_size': 64,
        'lr': 5e-3,
        'optim_type': 'SGD',
        'epochs': 100,
        'ckpt_path': './checkpoints',
        'dataset': 'fashionmnist',
        'model_type': 'joint_membership',

        'fusion_dim': 128,
        'n_random_pairs': 117,

        'num_runs': 5,

        'predict_run': 1,
        'predict_epoch': 100,
        'batch_timestamp': None,
    }

    if EXTERNAL_PARAMS:
        print(f"\n{'=' * 60}")
        print("Applying external parameter overrides:")
        for key, value in EXTERNAL_PARAMS.items():
            if key in base_config:
                old_value = base_config[key]
                base_config[key] = value
                print(f"  {key}: {old_value} → {value}")
            else:
                base_config[key] = value
                print(f"  {key}: {value} (new)")
        print(f"{'=' * 60}\n")

    config = Namespace(**base_config)

    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    my_logger.info(f"Experiment started at: {nowtime}")
    my_logger.info(f"Configuration: {config}")

    if train_flag:

        all_start = time.time()
        batch_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        all_run_best_acc = []
        all_run_best_prec = []
        all_run_best_rec = []
        all_run_best_f1 = []
        all_run_histories = []

        num_runs = getattr(config, 'num_runs', 1)

        for run_id in range(1, num_runs + 1):
            my_logger.info(f"\n{'=' * 70}")
            my_logger.info(f"║  RUN {run_id}/{num_runs}")
            my_logger.info(f"{'=' * 70}\n")

            run_start = time.time()

            model, history, best_acc_list, best_prec_list, best_rec_list, best_f1_list = trainModel(
                config,
                run_id=run_id,
                batch_timestamp=batch_timestamp
            )

            run_end = time.time()

            if len(best_acc_list) > 0:
                all_run_best_acc.append(best_acc_list[0])
                all_run_best_prec.append(best_prec_list[0])
                all_run_best_rec.append(best_rec_list[0])
                all_run_best_f1.append(best_f1_list[0])
                all_run_histories.append(history)

                my_logger.info(
                    f"Run {run_id}: Best Acc={best_acc_list[0]:.4f}%, "
                    f"Best F1={best_f1_list[0]:.4f}%, Time={run_end - run_start:.3f}s"
                )

            if len(history) > 0:
                plot_training_curves(
                    history,
                    config,
                    run_id=run_id,
                    batch_timestamp=batch_timestamp
                )

        my_logger.info("\n" + "=" * 70)
        my_logger.info("║  FINAL RESULTS (All Runs)")
        my_logger.info("=" * 70)
        my_logger.info(f"Total Runs: {len(all_run_best_acc)}")
        my_logger.info(f"Accuracy: {all_run_best_acc}")
        my_logger.info(f"Accuracy Stats: {cal_ms(all_run_best_acc)}")
        my_logger.info(f"Recall: {all_run_best_rec}")
        my_logger.info(f"Recall Stats: {cal_ms(all_run_best_rec)}")
        my_logger.info(f"Precision: {all_run_best_prec}")
        my_logger.info(f"Precision Stats: {cal_ms(all_run_best_prec)}")
        my_logger.info(f"F1: {all_run_best_f1}")
        my_logger.info(f"F1 Stats: {cal_ms(all_run_best_f1)}")
        my_logger.info("=" * 70)

        if len(all_run_histories) > 0:
            my_logger.info(f"\n{'=' * 60}")
            my_logger.info(f"Generating summary for {len(all_run_histories)} complete runs")
            my_logger.info(f"{'=' * 60}\n")

            plot_multi_run_summary(
                all_run_histories,
                config,
                batch_timestamp
            )

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
