import os

SLURM_TMPDIR = os.environ.get('SLURM_TMPDIR')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import torch
from random import shuffle
import dataloaders.base
from dataloaders.datasetGen import SplitGen, PermutedGen, RotatedGen
import methods
import numpy as np
import os
from tqdm import tqdm
import csv, copy
from torch.utils.data import DataLoader, Dataset
import sys
import datetime

result_root_path = r'./result'


# 构建超参数
def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0],
                        help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only", required=False)
    parser.add_argument('--run_seed', type=int, default=2023)
    parser.add_argument('--num_task', type=int, default=20)
    parser.add_argument('--categories', type=str,
                        default=['chair', 'sofa', 'airplane', 'bookshelf', 'bed', 'vase', 'monitor', 'table', 'toilet',
                                 'bottle', 'mantel', 'tv_stand', 'plant', 'piano', 'car', 'desk', 'dresser',
                                 'night_stand', 'glass_box', 'guitar', 'range_hood', 'bench', 'cone', 'tent',
                                 'flower_pot', 'laptop', 'keyboard', 'curtain', 'bathtub', 'sink', 'lamp', 'stairs',
                                 'door', 'radio', 'xbox', 'stool', 'person', 'wardrobe', 'cup', 'bowl'])
    # parser.add_argument('--categories', type=int,
    #                     default=['table', 'chair', 'airplane', 'car', 'file', 'stove', 'rocket', 'sofa', 'rifle', 'pillow', 'guitar', 'bag', 'lamp', 'telephone', 'earphone', 'clock', 'cabinet', 'pistol', 'cap', 'vessel', 'tower', 'laptop', 'bed', 'bottle', 'train', 'bookshelf', 'keyboard', 'monitor', 'microphone', 'washer', 'bus', 'jar'])
    parser.add_argument('--dataset', type=str, default='ShapeNet')
    # parser.add_argument('--dataroot', type=str, default='./pointCloud/data/shapenet.hdf5')
    parser.add_argument('--dataroot', type=str, default='./modelnet40_normal_resampled')
    parser.add_argument('--scale_mode', type=str, default='shape_unit',
                        help='global_unit, shape_unit, shape_bbox, shape_half, shape_34')
    parser.add_argument('--transform', type=str, default=None)
    parser.add_argument('--mem_size', type=int, default=500)
    parser.add_argument('--mem_batch', type=int, default=32)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--val_batch_size', type=int, default=64)
    parser.add_argument('--eps_mem_batch', type=int, default=16)

    args = parser.parse_args([])
    return args


args = init_args()


# 创建点云训练集和验证集
def ShapeNet(dataroot, train_categories, val_categories, scale_mode, transform):
    from pointCloud.utils.dataset import ShapeNetCore
    # 加载训练数据集
    train_dset = ShapeNetCore(
        path=dataroot,
        cates=train_categories,
        split='train',
        scale_mode=scale_mode,
        transform=transform,
    )
    # 加载验证数据集`
    val_dset = ShapeNetCore(
        path=dataroot,
        cates=val_categories,
        split='val',
        scale_mode=scale_mode,
        transform=transform,
    )

    return train_dset, val_dset


def ModelNet(dataroot, train_categories, val_categories, npoint=2048, normal_channels=False):
    from pointCloud.utils.dataset import ModelNetDataLoader
    train_dset = ModelNetDataLoader(root=dataroot, npoint=npoint, split='train', normal_channel=normal_channels)
    val_dset = ModelNetDataLoader(root=dataroot, npoint=npoint, split='test', normal_channel=normal_channels)
    # 创建一个列表，用于存储符合指定类别的数据点的索引
    train_selected_indices = []
    val_selected_indices = []
    for idx, (shape_name, _) in enumerate(train_dset.datapath):
        if shape_name in train_categories:
            train_selected_indices.append(idx)
    for idx, (shape_name, _) in enumerate(val_dset.datapath):
        if shape_name in val_categories:
            val_selected_indices.append(idx)

    train_class_subset_dataset = torch.utils.data.Subset(train_dset, train_selected_indices)
    val_class_subset_dataset = torch.utils.data.Subset(val_dset, val_selected_indices)
    return train_class_subset_dataset, val_class_subset_dataset


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


setup_seed(114514)

class RemapDataset(Dataset):
    def __init__(self, dataset, mapping):
        self.dataset = dataset
        self.mapping = mapping

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # ModelNetDataLoader returns (point_set, cls) where cls is np.array([cls])
        point_set, cls = self.dataset[idx]
        original_label = int(cls)
        if original_label in self.mapping:
            new_label = self.mapping[original_label]
            # Maintain the format: np.array([new_label])
            return point_set, np.array([new_label]).astype(np.int32)
        return point_set, cls

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # Ensure content is written immediately

    def flush(self):
        pass

if __name__ == '__main__':
    # Initialize logging
    log_dir = './CIL_logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file_name = f'{log_dir}/log_{current_time}.txt'
    sys.stdout = Logger(log_file_name)
    print(f"Logging to {log_file_name}")

    agent = None
    for n_task in range(0, args.num_task):
        train_now_categories = args.categories[n_task * 2: 2 + n_task * 2]
        # if n_task == 0:
        #     train_now_categories = args.categories[0: 2]  # 起始为二分类任务
        #
        # else:
        #     train_now_categories = args.categories[2 + n_task * 2: 4 + n_task * 2]  # 后续每次为一个二分类任务

        val_now_categories = args.categories[0: n_task * 2 + 2]
        # val_now_categories = args.categories[0: n_task * 2 + 4]

        outdim = 2 * n_task + 2  # 模型输出层神经元数量
        # outdim = 2 * (n_task + 2)  # 模型输出层神经元数量
        # train_dataset, val_dataset = ShapeNet(args.dataroot, train_now_categories, val_now_categories, args.scale_mode,
        #                                       args.transform)  # 创建数据集

        train_dataset, val_dataset = ModelNet(args.dataroot, train_now_categories, val_now_categories)
        
        # --- Label Remapping Logic ---
        # Get the original dataset to access category names to ID mapping
        # train_dataset is a Subset, so we access .dataset
        original_dataset = train_dataset.dataset 
        # original_dataset.classes is like {'airplane': 0, 'bathtub': 1, ...}
        
        # Create mapping: global_id -> incremental_id
        # The incremental order is defined by categories seen so far
        # val_now_categories contains all categories so far
        label_mapping = {}
        for incremental_id, cat_name in enumerate(val_now_categories):
            if cat_name in original_dataset.classes:
                global_id = original_dataset.classes[cat_name]
                label_mapping[global_id] = incremental_id
        
        print(f"Applying label mapping for Task {n_task}. Mapping size: {len(label_mapping)}")
        
        # Wrap datasets
        train_dataset = RemapDataset(train_dataset, label_mapping)
        val_dataset = RemapDataset(val_dataset, label_mapping)
        # -----------------------------

        if n_task == 0:
            agent = methods.MLP_correlation_CIL.MLP_correlation(args, outdim)  # 创建模型
            print(f"模型创建成功，当前输出层为{agent.model.out_dim}")
        else:
            agent.model.add_output_layer(2)  # 改变输出层神经元数量
            print(f"模型输出层改变成功，当前输出层为{agent.model.out_dim}")
        print(f'任务{n_task},类别为：{train_now_categories}')  # 输出当前任务类别

        dir_path = f'checkpoint/pointcloud/task_{n_task}/'
        if not os.path.exists(dir_path):  # 如果没有当前文件夹，则创建一个
            os.makedirs(dir_path)
        print(f'====================== Task {n_task} =======================')
        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, num_workers=8)
        if n_task == 0:
            val_original = copy.deepcopy(val_loader)
            Acc = agent.train(train_loader, val_loader, val_original, task_num=n_task)

        else:
            ref_model = copy.deepcopy(agent.model)  # 将当前model拷贝到ref_model上
            for p in ref_model.parameters():
                p.requires_grad = False  # 冻结ref_model的参数更新
            Acc = agent.train(train_loader, val_loader, val_original, task_num=n_task, ref_model=ref_model)
        print('task : {}  Acc : {}'.format(n_task, Acc))
