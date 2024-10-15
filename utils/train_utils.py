import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from models.Nets import MLP, CNNMnist, CNNCifar, VGG16, lenet, lenetMini, resnet20, STNet
from utils.sampling import iid, noniid, noniid_unbalanced
from .krum import KrumDefense

trans_mnist = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))])
trans_cifar10_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])
trans_cifar10_val = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
trans_cifar100_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                               std=[0.267, 0.256, 0.276])])
trans_cifar100_val = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                              std=[0.267, 0.256, 0.276])])

GTSRB_data_transforms = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])
trans_cifar10_train_aug = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomCrop(32, padding=4),  # 随机裁剪并填充
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 调整亮度、对比度等
    transforms.RandomRotation(15),  # 随机旋转
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])  # 归一化
])

from PIL import Image

class AugmentedDataset(Dataset):
    def __init__(self, concat_dataset, transform=None):
        self.concat_dataset = concat_dataset
        self.transform = transform
        self.data = []
        self.targets = []

        # 将所有子数据集的数据和标签整合
        for dataset in concat_dataset.datasets:
            self.data.extend(dataset.data)
            self.targets.extend(dataset.targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # 获取数据和对应的标签
        img, target = self.data[index], self.targets[index]

        # 将 numpy.ndarray 转换为 PIL.Image
        img = Image.fromarray(img)

        # 如果有定义 transform，则应用
        if self.transform:
            img = self.transform(img)

        return img, target



def get_data(args):
    if args.dataset == 'mnist' and args.ub_label !=  -1:
        dataset_train = datasets.MNIST(args.dataset_path+'/mnist', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST(args.dataset_path, train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users_train = iid(dataset_train, args.num_users)
            dict_users_test = iid(dataset_test, args.num_users)
        else:
            dict_users_train, rand_set_all = noniid_unbalanced(dataset_train, args.num_users, args.shard_per_user, ub_at=args.ub_label)
            dict_users_test, rand_set_all = noniid_unbalanced(dataset_test, args.num_users, args.shard_per_user, rand_set_all=rand_set_all, ub_at=args.ub_label)
            
    elif args.dataset == 'fmnist' and args.ub_label == -1:
        dataset_train = datasets.FashionMNIST(args.dataset_path+'/mnist', train=True, download=True, transform=transforms.ToTensor())
        dataset_test = datasets.FashionMNIST(args.dataset_path+'/mnist', train=False, download=True, transform=transforms.ToTensor())
        # sample users
        if args.iid:
            dict_users_train = iid(dataset_train, args.num_users)
            dict_users_test = iid(dataset_test, args.num_users)
        else:
            dict_users_train, rand_set_all = noniid(dataset_train, args.num_users, args.shard_per_user)
            dict_users_test, rand_set_all = noniid(dataset_test, args.num_users, args.shard_per_user, rand_set_all=rand_set_all)


    elif args.dataset == 'mnist' and args.ub_label == -1:
        dataset_train = datasets.MNIST(args.dataset_path+'/mnist', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST(args.dataset_path+'/mnist', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users_train = iid(dataset_train, args.num_users)
            dict_users_test = iid(dataset_test, args.num_users)
        else:
            dict_users_train, rand_set_all = noniid(dataset_train, args.num_users, args.shard_per_user)
            dict_users_test, rand_set_all = noniid(dataset_test, args.num_users, args.shard_per_user, rand_set_all=rand_set_all)
    
    elif args.dataset == 'cifar10' and not args.data_augmentation:
        dataset_train = datasets.CIFAR10(args.dataset_path+'/cifar10', train=True, download=True, transform=trans_cifar10_train)
        dataset_test = datasets.CIFAR10(args.dataset_path+'/cifar10', train=False, download=True, transform=trans_cifar10_val)
        if args.iid:
            dict_users_train = iid(dataset_train, args.num_users)
            dict_users_test = iid(dataset_test, args.num_users)
        else:
            dict_users_train, rand_set_all = noniid(dataset_train, args.num_users, args.shard_per_user)
            dict_users_test, rand_set_all = noniid(dataset_test, args.num_users, args.shard_per_user, rand_set_all=rand_set_all)
    
    elif args.dataset == 'cifar10' and args.data_augmentation:
        dataset_train = datasets.CIFAR10(args.dataset_path+'/cifar10', train=True, download=True, transform=trans_cifar10_train_aug)
        dataset_test = datasets.CIFAR10(args.dataset_path+'/cifar10', train=False, download=True, transform=trans_cifar10_val)
        augmented_dataset_list_train = [dataset_train for _ in range(args.data_augmentation)]
        dataset_train = AugmentedDataset(ConcatDataset(augmented_dataset_list_train), transform=trans_cifar10_train_aug)
        augmented_dataset_list_test = [dataset_test for _ in range(args.data_augmentation)]
        dataset_test = AugmentedDataset(ConcatDataset(augmented_dataset_list_test), transform=trans_cifar10_val)
        if args.iid:
            dict_users_train = iid(dataset_train, args.num_users)
            dict_users_test = iid(dataset_test, args.num_users)
        else:
            dict_users_train, rand_set_all = noniid(dataset_train, args.num_users, args.shard_per_user)
            dict_users_test, rand_set_all = noniid(dataset_test, args.num_users, args.shard_per_user, rand_set_all=rand_set_all)
    
    elif args.dataset == 'cifar100':
        dataset_train = datasets.CIFAR100(args.dataset_path+'/cifar100', train=True, download=True, transform=trans_cifar100_train)
        dataset_test = datasets.CIFAR100(args.dataset_path+'/cifar100', train=False, download=True, transform=trans_cifar100_val)
        if args.iid:
            dict_users_train = iid(dataset_train, args.num_users)
            dict_users_test = iid(dataset_test, args.num_users)
        else:
            dict_users_train, rand_set_all = noniid(dataset_train, args.num_users, args.shard_per_user)
            dict_users_test, rand_set_all = noniid(dataset_test, args.num_users, args.shard_per_user, rand_set_all=rand_set_all)
    elif args.dataset == 'GTSRB':
        dataset_train = datasets.GTSRB(root=dataset_path+"/GTSRB", split="train", transform=GTSRB_data_transforms, download=True)
        dataset_test = datasets.GTSRB(root=dataset_path+"/GTSRB", split="test", transform=GTSRB_data_transforms, download=True)
        if args.iid:
            dict_users_train = iid(dataset_train, args.num_users)
            dict_users_test = iid(dataset_test, args.num_users)
        else:
            dict_users_train, rand_set_all = noniid(dataset_train, args.num_users, args.shard_per_user)
            dict_users_test, rand_set_all = noniid(dataset_test, args.num_users, args.shard_per_user, rand_set_all=rand_set_all)
    else:
        exit('Error: unrecognized dataset')

    return dataset_train, dataset_test, dict_users_train, dict_users_test

def get_model(args):
    if args.model == 'cnn' and args.dataset in ['cifar10', 'cifar100']:
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'vgg' and args.dataset == 'cifar10':
        net_glob = VGG16(args=args).to(args.device)
    elif args.model == 'resnet20' and args.dataset == 'cifar10':
        net_glob = resnet20(args=args).to(args.device)
    elif args.model == 'lenet' and args.dataset == 'mnist':
        net_glob = lenet(args=args).to(args.device)
    elif args.model == 'lenetMini' and args.dataset == 'mnist':
        net_glob = lenetMini(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp' and args.dataset == 'mnist':
        net_glob = MLP(dim_in=784, dim_hidden=256, dim_out=args.num_classes).to(args.device)
    elif args.model == 'lenet' and args.dataset == 'fmnist':
        net_glob = lenet(args=args).to(args.device)
    elif args.model == 'stn' and args.dataset == 'GTSRB':
        net_glob = STNet(args=args).to(args.device)
    elif args.model == 'resnet20' and args.dataset == 'GTSRB':
        net_glob = resnet20(args=args).to(args.device)
    else:
        exit('Error: unrecognized model')

    return net_glob

def getWglob(w_glob_list: list):
    w = w_glob_list[0][1]
    if (len(w_glob_list) <= 1):
        return w
        
    user_weight = w_glob_list[0][2]
    for k in w.keys():
        w[k] *= w_glob_list[0][2]

    for idx, w_local, idxs_weight in w_glob_list[1:]:
        if idxs_weight < 10:
            continue
        # w += w_local * idxs_weight
        user_weight += idxs_weight
        for k in w.keys():
            w[k] += w_local[k] * idxs_weight

    for k in w.keys():
        w[k] = torch.div(w[k], user_weight)

    return w

def getWglobKrum(w_glob_list: list, krumClients=70, mclients=3):
    kd = KrumDefense(mclients, krumClients)
    clients = []
    for idx, w_local, idxs_weight in w_glob_list:
        clients.append(tuple([idxs_weight, w_local]))
    clients = kd.defend_before_aggregation(clients)

    print(len(clients))

    w = clients[0][1]
    user_weight = clients[0][0]
    for k in w.keys():
        w[k] *= w_glob_list[0][0]

    for idxs_weight, w_local in clients[1:]:
        # w += w_local * idxs_weight
        user_weight += idxs_weight
        for k in w.keys():
            w[k] += w_local[k] * idxs_weight

    for k in w.keys():
        w[k] = torch.div(w[k], user_weight)

    return w