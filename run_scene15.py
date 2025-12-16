import torch
import os
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
from data.scene15.data_loader import load_dataset_scene15
from data.dirtyMnist.data_loader import load_dataset_dmnist
from data.cifar10.data_reader import load_dataset_cifar10
from utils.evaluation import cal_ms, cal_merics
from models.My_plot import plot_training_curves, plot_multi_run_summary
import inspect

MyNetwork_cifar10_noent_simply = None
MyNetwork_cifar10_ent_simply = None
MyNetwork_cifar10_noent = None
MyNetwork_cifar10_ent = None
cifar10_joint_membership = None
DirtyMnist_JointMembership = None

try:
    from models.My_conditional_model import (
        MyNetwork_cifar10_2Qubit as MyNetwork_cifar10_noent_simply,
        Cifar10_singleLayer_ent as MyNetwork_cifar10_ent_simply,
        Cifar10_doubleLayer_noEnt as MyNetwork_cifar10_noent,
        MyNetwork_cifar10_2Qubit_2Layer as MyNetwork_cifar10_ent,
    )

    print("✓ Loaded: CIFAR-10 conditional models from My_conditional_model.py")
    print("  - double_noent_simply: MyNetwork_cifar10_2Qubit")
    print("  - double_ent_simply: Cifar10_singleLayer_ent")
    print("  - double_noent: Cifar10_doubleLayer_noEnt")
    print("  - double_ent: MyNetwork_cifar10_2Qubit_2Layer")
except ImportError as e:
    print(f"Warning: My_conditional_model.py not found - {e}")

try:
    from models.My_joint_cifar import Cifar10_JointMembership as cifar10_joint_membership

    print("✓ Loaded: CIFAR-10 joint membership model from My_joint_cifar.py")
    print("  - joint_membership: Cifar10_JointMembership")
except ImportError as e:
    print(f"Warning: My_joint_cifar.py not found - {e}")

try:
    from models.My_joint_dmnist import DirtyMnist_JointMembership

    print("✓ Loaded: Dirty MNIST joint membership model from My_joint_dmnist.py")
    print("  - joint_membership (dmnist): DirtyMnist_JointMembership")
except ImportError as e:
    print(f"Warning: My_joint_dmnist.py not found - {e}")

try:
    from models.My_joint_jaffe import JAFFE_JointMembership

    print("✓ Loaded: JAFFE joint membership model from My_joint_jaffe.py")
    print("  - joint_membership (jaffe): JAFFE_JointMembership")
except ImportError as e:
    print(f"Warning: My_joint_jaffe.py not found - {e}")

try:
    from models.My_joint_scene15 import SCENE15_JointMembership

    print("✓ Loaded: Scene15 model from My_joint_scene15.py")
    print("  - Scene15 DNN/FDNN/QA-HFNN: SCENE15_JointMembership")
except ImportError as e:
    print(f"Warning: My_joint_scene15.py not found - {e}")

MyNetwork_scene15_noent_simply = None
MyNetwork_scene15_ent_simply = None
MyNetwork_scene15_noent = None
MyNetwork_scene15_ent = None

MyNetwork_dmnist_noent_simply = None
MyNetwork_dmnist_ent_simply = None
MyNetwork_dmnist_noent = None
MyNetwork_dmnist_ent = None

MyNetwork_JAFFE_noent_simply = None
MyNetwork_JAFFE_ent_simply = None
MyNetwork_JAFFE_noent = None
MyNetwork_JAFFE_ent = None

import torch.optim as optim
import datetime
import time
import logging
import sys


def load_external_params():
    import json
    params_json = os.environ.get('SCENE15_TRAIN_PARAMS')
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


# Initialize
my_logger = init_logger("./run_result_2qubit.log", "ml_logger_2qubit")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_and_dataset(config, dataset_type, model_type):
    """
    Returns the corresponding model and data based on dataset type and model type

    Parameters:
        dataset_type: Dataset type ('scene15', 'cifar10', 'dmnist')
        model_type: Model type
            - 'double_noent_simply': Two-qubit non-entangled single-layer simplified version
            - 'double_ent_simply': Two-qubit entangled single-layer simplified version
            - 'double_noent': Two-qubit non-entangled full version
            - 'double_ent': Two-qubit entangled full version
            - 'joint_membership': Joint membership model
    """

    if dataset_type == 'scene15':
        data_train, data_test = load_dataset_scene15()
        num_classes = 15
    elif dataset_type == 'cifar10':

        import torchvision.transforms as transforms

        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
        ])

        data_train, data_test = load_dataset_cifar10(
            train_transform=train_transform,
            test_transform=test_transform
        )
        num_classes = 10
    elif dataset_type == 'dmnist':
        # Use default data augmentation configuration from data_loader.py
        # RandomRotation: Images are randomly rotated within ±10 degrees
        # RandomAffine (translation): Images are randomly shifted by up to 10% of width/height
        # RandomAffine (scaling): Images are randomly scaled between 90% and 110% of original size
        data_train, data_test = load_dataset_dmnist()
        num_classes = 10
    elif dataset_type == 'jaffe':
        from data.jaffe.data_loader import load_dataset_jaffe
        if not (hasattr(config, 'k') and hasattr(config, 'fold')):
            raise ValueError(
                "JAFFE dataset requires K-fold configuration.\n"
                "Please set config.k (e.g., k=8) and config.fold (0 to k-1) in your config."
            )

        k = config.k
        fold = config.fold
        data_train, data_test = load_dataset_jaffe(
            k=k,
            fold=fold,
            image_size=32
        )
        print(f"✅ JAFFE K-fold mode: k={k}, fold={fold}")
        num_classes = 7
    else:
        raise ValueError(f"Unsupported dataset: {dataset_type}")
    model_mapping = {
        'scene15': {
            'double_noent_simply': MyNetwork_scene15_noent_simply,
            'double_ent_simply': MyNetwork_scene15_ent_simply,
            'double_noent': MyNetwork_scene15_noent,
            'double_ent': MyNetwork_scene15_ent,
            'joint_membership': SCENE15_JointMembership,
        },
        'cifar10': {
            'double_noent_simply': MyNetwork_cifar10_noent_simply,
            'double_ent_simply': MyNetwork_cifar10_ent_simply,
            'double_noent': MyNetwork_cifar10_noent,
            'double_ent': MyNetwork_cifar10_ent,
            'joint_membership': cifar10_joint_membership,
        },
        'dmnist': {
            'double_noent_simply': MyNetwork_dmnist_noent_simply,
            'double_ent_simply': MyNetwork_dmnist_ent_simply,
            'double_noent': MyNetwork_dmnist_noent,
            'double_ent': MyNetwork_dmnist_ent,
            'joint_membership': DirtyMnist_JointMembership,
        },
        'jaffe': {
            'double_noent_simply': MyNetwork_JAFFE_noent_simply,
            'double_ent_simply': MyNetwork_JAFFE_ent_simply,
            'double_noent': MyNetwork_JAFFE_noent,
            'double_ent': MyNetwork_JAFFE_ent,
            'joint_membership': JAFFE_JointMembership,
        },

    }

    model_class = model_mapping.get(dataset_type, {}).get(model_type)

    if model_class is None:
        available_models = []
        for ds, models in model_mapping.items():
            for mt, mc in models.items():
                if mc is not None:
                    available_models.append(f"{ds}/{mt}")

        error_msg = f"\nModel '{model_type}' for dataset '{dataset_type}' not found or failed to import."
        error_msg += f"\n\nAvailable models:"
        if available_models:
            for am in available_models:
                error_msg += f"\n  - {am}"
        else:
            error_msg += "\n  No models were successfully imported!"
            error_msg += "\n\nPlease check:"
            error_msg += "\n  1. Files exist: My_conditional_model.py, My_joint_cifar.py, My_joint_dmnist.py"
            error_msg += "\n  2. Class names are correct"
            error_msg += "\n  3. No import errors in the model files"

        raise ImportError(error_msg)

    return data_train, data_test, model_class, num_classes


def trainModel(config, run_id=None, batch_timestamp=None):
    # 获取数据集类型和模型类型
    dataset_type = getattr(config, 'dataset', 'cifar10')
    model_type = getattr(config, 'model_type', 'double_noent_simply')

    if batch_timestamp and run_id:

        if hasattr(config, 'fold'):

            checkpoints_path = os.path.join(
                config.ckpt_path,
                dataset_type,
                model_type,
                f'batch_{batch_timestamp}',
                f'run_{run_id}',

            )
        else:

            checkpoints_path = os.path.join(
                config.ckpt_path,
                dataset_type,
                model_type,
                f'batch_{batch_timestamp}',
                f'run_{run_id}'
            )
    else:

        checkpoints_path = os.path.join(config.ckpt_path, dataset_type, model_type)

    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path, exist_ok=True)
        print(f"{checkpoints_path} is created")

    batchsize = config.batch_size
    epochs = config.epochs
    LR = config.lr

    data_train, data_test, model_class, num_classes = get_model_and_dataset(config, dataset_type, model_type)

    my_logger.info(f"Using dataset: {dataset_type}, model_type: {model_type}, num_classes: {num_classes}")

    train_data_size = len(data_train)
    valid_data_size = len(data_test)
    print(
        f'Dataset: {dataset_type}, Model: {model_type}, train_size: {train_data_size:4d}, valid_size: {valid_data_size:4d}')

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

    history = []
    best_acc_list = []
    best_prec_list = []
    best_rec_list = []
    best_f1_list = []

    if model_type == 'joint_membership':

        if dataset_type == 'dmnist':
            fusion_dim = getattr(config, 'fusion_dim', 128)
            n_random_pairs = getattr(config, 'n_random_pairs', 3)
            model = model_class(
                num_classes=num_classes,
                fusion_dim=fusion_dim,
                n_random_pairs=n_random_pairs
            ).to(device)
        elif dataset_type == 'jaffe':
            hidden_dim = getattr(config, 'hidden_dim', 512)
            n_qnn_layers = getattr(config, 'n_qnn_layers', 3)
            model = model_class(
                num_classes=num_classes,
                hidden_dim=hidden_dim,
                n_qnn_layers=n_qnn_layers
            ).to(device)
        elif dataset_type == 'cifar10':
            reduced_dim = getattr(config, 'reduced_dim', 16)
            n_adjacent_pairs = getattr(config, 'n_adjacent_pairs', None)
            n_random_pairs = getattr(config, 'n_random_pairs', 2)
            model = model_class(
                num_classes=num_classes,
                hidden_dim=256,
                reduced_dim=reduced_dim,
                n_adjacent_pairs=n_adjacent_pairs,
                n_random_pairs=n_random_pairs
            ).to(device)
        elif dataset_type == 'scene15':

            hidden_dim = getattr(config, 'hidden_dim', 256)
            n_qnn_layers = getattr(config, 'n_qnn_layers', 1)
            n_random_pairs = getattr(config, 'n_random_pairs', 0)

            n_pairs = getattr(config, 'n_pairs', None)
            top_k = getattr(config, 'top_k', 50)
            bottom_m = getattr(config, 'bottom_m', 50)

            import inspect
            model_params = inspect.signature(model_class.__init__).parameters

            kwargs = {
                'num_classes': num_classes,
                'hidden_dim': hidden_dim,
                'n_qnn_layers': n_qnn_layers,
                'n_random_pairs': n_random_pairs
            }

            if 'n_pairs' in model_params:
                kwargs['n_pairs'] = n_pairs
            if 'top_k' in model_params:
                kwargs['top_k'] = top_k
            if 'bottom_m' in model_params:
                kwargs['bottom_m'] = bottom_m
            if 'median_range' in model_params:
                kwargs['median_range'] = getattr(config, 'median_range', (80, 120))
            if 'pair_ratio' in model_params:
                pair_ratio = getattr(config, 'pair_ratio', None)
                if pair_ratio is not None:

                    if isinstance(pair_ratio, list):
                        kwargs['pair_ratio'] = tuple(pair_ratio)
                    else:
                        kwargs['pair_ratio'] = pair_ratio

            model = model_class(**kwargs).to(device)
        else:

            model = model_class(num_classes=num_classes).to(device)
    else:

        if dataset_type == 'scene15':
            model = model_class(class_num=num_classes).to(device)
        elif dataset_type in ['cifar10', 'dmnist', 'jaffe']:

            hidden_dim = getattr(config, 'hidden_dim', 256)
            model = model_class(num_classes=num_classes, hidden_dim=hidden_dim).to(device)
        else:

            model = model_class(num_classes=num_classes).to(device)
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

    my_logger.info(f"Initial learning rate: {LR}")
    # # Use cosine annealing learning rate scheduler
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer=optimizer,
    #     T_max=epochs,
    #     eta_min=1e-5  # Minimum learning rate
    # )
    # my_logger.info(f"Using CosineAnnealingLR: T_max={epochs}, eta_min=1e-5")
    # Using step decay instead
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        milestones=[int(epochs * 0.56), int(epochs * 0.78)],
        gamma=0.1,
        last_epoch=-1
    )
    my_logger.info(f"Using MultiStepLR: milestones=[{int(epochs * 0.56)}, {int(epochs * 0.78)}], gamma=0.1")
    # Cosine annealing with warmup
    # my_logger.info(f"Initial learning rate: {LR}")
    #
    # # Warm-up + CosineAnnealingLR
    # warmup_epochs = getattr(config, 'warmup_epochs', 0)  # Disabled by default (backward compatible)
    #
    # if warmup_epochs > 0:
    #     # Custom Lambda scheduler implementing Warm-up + Cosine
    #     def warmup_cosine_lambda(epoch):
    #         if epoch < warmup_epochs:
    #             # Warm-up phase: linear increase from 0 to 1
    #             return (epoch + 1) / warmup_epochs
    #         else:
    #             # Cosine phase: decay from 1 to eta_min/LR
    #             progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
    #             min_lr_ratio = 1e-5 / LR  # Ratio of eta_min to initial learning rate
    #             return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + np.cos(np.pi * progress))
    #
    #     scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_lambda)
    #     my_logger.info(f"Using Warm-up ({warmup_epochs} epochs) + CosineAnnealingLR (eta_min=1e-5)")
    # else:
    #     # Without Warm-up (original CosineAnnealingLR)
    #     scheduler = optim.lr_scheduler.CosineAnnealingLR(
    #         optimizer=optimizer,
    #         T_max=epochs,
    #         eta_min=1e-5
    #     )
    #     my_logger.info(f"Using CosineAnnealingLR: T_max={epochs}, eta_min=1e-5")

    best_acc = 0
    best_epoch = 0
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

        # torch.save(model, os.path.join(checkpoints_path, f'{epoch + 1:02d}.pt'))

    best_prec = precision_t[best_epoch - 1]
    best_rec = recall_t[best_epoch - 1]
    best_f1 = f1_t[best_epoch - 1]

    best_prec_list.append(best_prec * 100)
    best_rec_list.append(best_rec * 100)
    best_f1_list.append(best_f1 * 100)
    best_acc_list.append(best_acc * 100)

    try:
        model.load_state_dict(best_model_state)
        if hasattr(config, 'fold'):
            model_filename = f'best_fold{config.fold}.pt'
        else:
            model_filename = f'best_epoch{best_epoch:02d}.pt'

        model_save_path = os.path.join(checkpoints_path, model_filename)
        torch.save(model, model_save_path)
        my_logger.info(f"\n✓ Final best model saved: {model_filename} (Acc={best_acc:.4f}, Epoch={best_epoch})")
    except Exception as e:
        my_logger.error(f"\n✗ Failed to save final best model: {e}")

    history = np.array(history)
    return model, history, best_acc_list, best_prec_list, best_rec_list, best_f1_list


def PredictModel(config):
    batchsize = config.batch_size
    dataset_type = getattr(config, 'dataset', 'cifar10')
    model_type = getattr(config, 'model_type', 'double_noent_simply')

    run_num = getattr(config, 'predict_run', 1)
    epoch_num = getattr(config, 'predict_epoch', 100)
    fold_num = getattr(config, 'predict_fold', None)

    data_train, data_test, _, _ = get_model_and_dataset(config, dataset_type, model_type)

    if fold_num is not None:
        my_logger.info(
            f"PredictModel using dataset: {dataset_type}, model: {model_type}, "
            f"run: {run_num}, fold: {fold_num}")
    else:
        my_logger.info(
            f"PredictModel using dataset: {dataset_type}, model: {model_type}, "
            f"run: {run_num}, epoch: {epoch_num}")

    valid_data_size = len(data_test)
    test_loader = DataLoader(data_test, batch_size=batchsize, shuffle=False)

    batch_timestamp = getattr(config, 'batch_timestamp', None)

    if batch_timestamp:

        base_path = os.path.join('./checkpoints', dataset_type, model_type,
                                 f'batch_{batch_timestamp}', f'run_{run_num}')
    else:

        base_path = os.path.join('./checkpoints', dataset_type, model_type, f'run_{run_num}')

    if fold_num is not None:

        model_path = os.path.join(base_path, f'best_fold{fold_num}.pt')
        my_logger.info(f"Loading K-fold model: fold={fold_num}")
    else:

        model_path = os.path.join(base_path, f'best_epoch{epoch_num:02d}.pt')
        my_logger.info(f"Loading standard model: epoch={epoch_num}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Please check:\n"
            f"  - dataset={dataset_type}, model={model_type}\n"
            f"  - run_num={run_num}, fold_num={fold_num}, epoch_num={epoch_num}\n"
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
        'project_name': 'scene15_dnn_baseline',
        'batch_size': 32,
        'lr': 0.01,
        'optim_type': 'SGD',
        'warmup_epochs': 0,
        'epochs': 100,
        'ckpt_path': './checkpoints',

        'dataset': 'scene15',
        'model_type': 'joint_membership',

        'hidden_dim': 256,
        'num_runs': 10,

        # ===== Dual-Qubit Pairing Parameters (DQHFNN only) =====
        'n_pairs': 50,  # Total number of pairs
        'top_k': 40,  # Number of strongest dimensions
        'bottom_m': 40,  # Number of weakest dimensions
        'median_range': (80, 120),  # Median range (tuple format)

        # ===== Pairing Ratio (optional, choose one of three) =====
        # Method 1: Use ratio (recommended)
        'pair_ratio': (0.6, 0.4),  # (Strongest↔Weakest%, Strongest↔Median%) Must sum to 1.0

        # Method 2: Direct quantity specification (if not using ratio)
        # 'n_top_bottom_pairs': 30,
        # 'n_top_median_pairs': 20,

        # Method 3: No specification (uses default 50%-50%)
        # Comment out pair_ratio
        # Prediction parameters
        'predict_run': 40,
        'predict_epoch': 2,
        'predict_fold': None,
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
    # ==================== End of Scene15 Configuration ====================
    # # JAFFE + K-fold mode
    # config = Namespace(
    #     project_name='2qubit_quantum_nopoolmax',
    #     batch_size=8,
    #     lr=5e-4,  # ✅ Maintain original learning rate
    #     optim_type='AdamW',  # ✅ Upgraded optimizer
    #     warmup_epochs=3,  # ✅ Added mild Warm-up
    #     epochs=100,
    #     ckpt_path='./checkpoints',
    #
    #     dataset='jaffe',
    #     model_type='joint_membership',
    #
    #     hidden_dim=128,
    #     n_qnn_layers=3,
    #     k=8,
    #     # Random pairs n_random_pairs defaults to 48 (unused parameter)
    #     num_runs=5,
    #
    #     #====Prediction parameters==========
    #     predict_run=1,
    #     predict_epoch=2,
    #     predict_fold=None,
    #     batch_timestamp=None,
    # )

    # # ========== Example Configurations for Other Datasets (uncomment to use) ==========
    # # DMNIST + joint_membership
    # config = Namespace(
    #     project_name='2qubit_quantum_fuzzy',
    #     batch_size=64,
    #     lr=5e-3,
    #     optim_type='SGD',
    #     epochs=100,
    #     ckpt_path='./checkpoints',
    #     dataset='dmnist',
    #     model_type='joint_membership',
    #     fusion_dim=128,
    #     n_random_pairs=48,
    #     num_runs=1,
    # )

    # # CIFAR-10 + joint_membership
    # config = Namespace(
    #     project_name='2qubit_quantum_fuzzy',
    #     batch_size=64,
    #     lr=1e-3,
    #     optim_type='Adam',
    #     epochs=150,
    #     ckpt_path='./checkpoints',
    #     dataset='cifar10',
    #     model_type='joint_membership',
    #     reduced_dim=16,
    #     n_random_pairs=2,
    #     num_runs=1,
    # )

    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    my_logger.info(f"Experiment started at: {nowtime}")
    my_logger.info(f"Configuration: {config}")

    if train_flag:
        all_start = time.time()
        batch_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        dataset_type = getattr(config, 'dataset', 'cifar10')
        use_kfold = False
        k_folds = 1
        """
        getattr(config, 'dataset', 'cifar10') means:

        Attempt to get the dataset attribute from config object
        If config.dataset exists, return its value (e.g., 'jaffe', 'dmnist', 'cifar10')
        If config.dataset doesn't exist, return default value 'cifar10'

        If not in K-fold mode (standard mode), k_folds = 1 means training runs only once (no folds)
        """

        if dataset_type == 'jaffe' and hasattr(config, 'k'):
            use_kfold = True
            k_folds = config.k
            my_logger.info(f"✅ K-Fold Cross-Validation Enabled: {k_folds} folds")
        else:
            my_logger.info(f"✅ Standard Training Mode (No K-Fold)")

        all_run_best_acc = []
        all_run_best_prec = []
        all_run_best_rec = []
        all_run_best_f1 = []

        if use_kfold:
            all_fold_best_acc = []
            all_fold_best_prec = []
            all_fold_best_rec = []
            all_fold_best_f1 = []

        all_run_histories = []

        num_runs = getattr(config, 'num_runs', 1)

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
                if use_kfold:
                    config.fold = fold_idx
                    my_logger.info(f"\n--- Fold {fold_idx + 1}/{k_folds} ---")

                # 单次训练
                model, fold_history, best_acc_list, best_prec_list, best_rec_list, best_f1_list = trainModel(
                    config,
                    run_id=run_id,
                    batch_timestamp=batch_timestamp
                )

                run_fold_best_acc.extend(best_acc_list)
                run_fold_best_prec.extend(best_prec_list)
                run_fold_best_rec.extend(best_rec_list)
                run_fold_best_f1.extend(best_f1_list)

                if use_kfold and len(best_acc_list) > 0:
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

                if use_kfold:

                    all_fold_best_acc.extend(run_fold_best_acc)
                    all_fold_best_prec.extend(run_fold_best_prec)
                    all_fold_best_rec.extend(run_fold_best_rec)
                    all_fold_best_f1.extend(run_fold_best_f1)
                else:

                    all_run_best_acc.append(run_mean_acc)
                    all_run_best_prec.append(run_mean_prec)
                    all_run_best_rec.append(run_mean_rec)
                    all_run_best_f1.append(run_mean_f1)

                if use_kfold:
                    my_logger.info(f"\n{'=' * 70}")
                    my_logger.info(f"║  Run {run_id} Summary (Averaged over {k_folds} folds)")
                    my_logger.info(f"{'=' * 70}")
                    my_logger.info(f"Mean Accuracy:  {run_mean_acc:.4f}% ± {np.std(run_fold_best_acc):.4f}%")
                    my_logger.info(f"Mean Precision: {run_mean_prec:.4f}% ± {np.std(run_fold_best_prec):.4f}%")
                    my_logger.info(f"Mean Recall:    {run_mean_rec:.4f}% ± {np.std(run_fold_best_rec):.4f}%")
                    my_logger.info(f"Mean F1:        {run_mean_f1:.4f}% ± {np.std(run_fold_best_f1):.4f}%")
                    my_logger.info(f"Time: {run_end - run_start:.2f}s")
                    my_logger.info(f"{'=' * 70}\n")
                else:
                    my_logger.info(
                        f"Run {run_id}: Best Acc={run_mean_acc:.4f}%, Best F1={run_mean_f1:.4f}%, Time={run_end - run_start:.3f}s")

            if len(run_fold_histories) > 0:
                if use_kfold:

                    if len(run_fold_histories) < k_folds:
                        my_logger.warning(
                            f"⚠️  Only {len(run_fold_histories)}/{k_folds} folds have complete histories")

                    fold_lengths = [len(h) for h in run_fold_histories]
                    min_length = min(fold_lengths)
                    max_length = max(fold_lengths)

                    if min_length != max_length:
                        my_logger.warning(
                            f"⚠️  K-fold histories have different lengths: min={min_length}, max={max_length}")
                        my_logger.warning(f"   Truncating all histories to {min_length} epochs for averaging")
                        run_fold_histories = [h[:min_length] for h in run_fold_histories]

                    try:

                        for i, h in enumerate(run_fold_histories):
                            if h.ndim != 2 or h.shape[1] != 4:
                                raise ValueError(f"Fold {i + 1} has invalid shape: {h.shape}")

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
                        my_logger.error(f"   Fold history shapes: {[h.shape for h in run_fold_histories]}")
                else:

                    plot_training_curves(
                        run_fold_histories[0],
                        config,
                        run_id=run_id,
                        batch_timestamp=batch_timestamp
                    )

                    all_run_histories.append(run_fold_histories[0])

        my_logger.info("\n" + "=" * 70)

        if use_kfold:

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
        else:

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

        if not use_kfold and len(all_run_histories) > 0:
            my_logger.info(f"\n{'=' * 60}")
            my_logger.info(f"Generating summary for {len(all_run_histories)} complete runs")
            my_logger.info(f"{'=' * 60}\n")

            plot_multi_run_summary(
                all_run_histories,
                config,
                batch_timestamp
            )
        elif use_kfold:
            my_logger.info("\nℹ️  K-Fold mode: Summary plots across runs are not generated")
            my_logger.info("   (Each run already represents an average over multiple folds)")

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
