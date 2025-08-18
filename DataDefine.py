import numpy as np
from torch.utils.data import Dataset
from utils import print_log, seed_worker, get_scaler, add_gaussian_noise
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def get_datloader(args, mode = "train", infer_num = [-1]):
    """Generate dataloader"""
    after_num = args.data_after_num + 1
    if "AE" in args.model_type:
        dataset = AE_Dataset(args)
    elif "DON" in args.model_type:
        dataset = DON_Dataset(args)
        after_num=1

    data_scaler = dataset.scaler if dataset.scaler else None

    if mode == "inference":
        if infer_num == [-1]:
            infer_num = range(len(dataset))
        inference_dataset = Subset(dataset, infer_num) 
        try:
            orgs_data = dataset.orgs[infer_num]
        except:
            orgs_data = None
        infer_loader = DataLoader(inference_dataset,
                                num_workers=args.num_workers,
                                batch_size=args.per_device_valid_batch_size,
                                pin_memory=True,
                                shuffle = False,
                                drop_last=False)
        print_log(f"Length of inference_dataset: {len(inference_dataset)}")
        print_log(f"Shape of input_data: {inference_dataset[0][0].shape}")
        print_log(f"Shape of label_data: {inference_dataset[0][1].shape}")
        return infer_loader, data_scaler, dataset.x_mesh, dataset.y_mesh, orgs_data
    
    # Split dataset into training dataset, validation dataset and test_dataset

    if (args.test_ratio > 1) or (args.test_ratio < 0):
        print_log(f"Errot test_ratio!")
        raise EOFError

    testlen = int(args.test_ratio * args.data_num) - 1 
    validlen = int(args.valid_ratio * args.data_num)
    trainlen = args.data_num - 1 - testlen - validlen

    sample_indices = np.random.permutation(args.data_num - 1)
    train_indices = sample_indices[:trainlen]
    valid_indices = sample_indices[trainlen:trainlen+validlen]
    test_indices = sample_indices[-testlen:]
    test_indices = np.append(test_indices, args.data_num-1)

    if args.train_ratio < (1-args.test_ratio):
        print_log("Use less trainlen")
        testlen_act = int(args.train_ratio * args.data_num)
        train_generator = np.random.default_rng(args.train_seed)
        train_indices = train_generator.choice(train_indices, testlen_act,
                                               replace=False, shuffle=False)

    def act_indices(indices, t):
        act_indices = []
        for idx in indices:
            act_indices.extend(range(idx*t, (idx+1)*t))
        return np.array(act_indices)
    
    train_indices_act = act_indices(train_indices, after_num)
    test_indices_act = act_indices(test_indices, after_num)
    if args.valid_ratio > 0:
        valid_indices_act = act_indices(valid_indices, after_num)
    else:
        valid_indices_act = test_indices_act

    train_dataset = Subset(dataset,train_indices_act)
    valid_dataset = Subset(dataset,valid_indices_act)
    test_dataset = Subset(dataset,test_indices_act)

    try:
        orgs_data = dataset.orgs[test_indices_act]
    except:
        orgs_data = None

    if args.init:
        print_log(f"Length of all dataset: {len(dataset)}")
        print_log(f"Length of train_dataset: {len(train_dataset)}")
        print_log(f"Length of valid_dataset: {len(valid_dataset)}")
        print_log(f"Length of test_dataset: {len(test_dataset)}")
        print_log(f"Shape of input_data: {test_dataset[0][0].shape}")
        print_log(f"Shape of label_data: {test_dataset[0][1].shape}")

    # DataLoaders creation:
    if not args.dist:
        train_sampler = RandomSampler(train_dataset)
        vaild_sampler = SequentialSampler(valid_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        vaild_sampler = DistributedSampler(valid_dataset)

    train_loader = DataLoader(train_dataset,
                                sampler=train_sampler,
                                num_workers=args.num_workers,
                                drop_last=args.drop_last,
                                pin_memory=True,
                                batch_size=args.per_device_train_batch_size,
                                worker_init_fn=seed_worker)
    vali_loader = DataLoader(valid_dataset,
                                sampler=vaild_sampler,
                                num_workers=args.num_workers,
                                drop_last=False,
                                pin_memory=True,
                                batch_size=args.per_device_valid_batch_size,
                                worker_init_fn=seed_worker)
    test_loader = DataLoader(test_dataset,
                                num_workers=args.num_workers,
                                pin_memory=True,
                                batch_size=args.per_device_valid_batch_size,
                                shuffle = False,
                                drop_last=False,
                                worker_init_fn=seed_worker)
    return train_loader, vali_loader, test_loader, data_scaler, dataset.x_mesh, dataset.y_mesh, orgs_data


def get_lam_datloader(args, inputs, labels, scaler):
    dataset = lam_Dataset(inputs, labels, scaler)
    loader = DataLoader(dataset,
                                num_workers=args.num_workers,
                                pin_memory=True,
                                batch_size=args.per_device_valid_batch_size,
                                shuffle = False,
                                drop_last=False)
    return loader

class AE_Dataset(Dataset):
    ''' Dataset for loading and preprocessing the AE data '''
    def __init__(self, args):
        # Read inputs
        self.features, self.height, self.width = args.data_shape
        self.data_all = [0] + args.data_after
        self.data_after_num = args.data_after_num
        self.data_num = args.data_num
        self.features = args.features
        self.data_select = args.data_select
        mesh = np.load(args.mesh_path)
        data = np.load(args.org_path)
        data_matrix = data[:self.data_num, self.data_all].astype(np.float64).copy()
        data_matrix = data_matrix[:, :, self.data_select] 
        
        self.y_mesh, self.x_mesh = mesh[0], mesh[1]

        # Data Standard
        self.scaler = [[]]
        for i in range(self.features):
            self.scaler[0].append(get_scaler(args, 0, i))

        # Reshape data: (num, data_time, features, height, width) -> (num * data_time, features, height * width)
        data_matrix = data_matrix.reshape(self.data_num * (self.data_after_num+1), self.features, -1)
        
        noise_matrix = add_gaussian_noise(data_matrix, args.noise_scale, args.sample_ratio,
                                         args.mask_ratio, args.samlpes_seed, args.mask_seed, args.noise_seed)
        label_data = data_matrix.copy()

        for i in range(self.features):
            noise_matrix[:,i] = self.scaler[0][i].transform(noise_matrix[:,i])
            label_data[:,i] = self.scaler[0][i].transform(label_data[:,i])

        # (num * data_time, 1, features, height, width)
        self.data_matrix = label_data.reshape(self.data_num * (self.data_after_num+1), 1, 
                                              self.features, self.height, self.width) 
        
        self.noise_matrix = noise_matrix.reshape(self.data_num * (self.data_after_num+1), 1,
                                                 self.features, self.height, self.width)
        del mesh
        del label_data, data_matrix


    def __getitem__(self, index):
        # Convert Data into PyTorch tensors
        inputs = self.noise_matrix[index]        #(num, 1, feature, height, width)
        labels = self.data_matrix[index]        #(num, 1, feature, height, width)
        inputs = torch.Tensor(inputs)
        labels = torch.Tensor(labels)
        return inputs, labels


    def __len__(self):
        # Returns the size of the dataset
        return self.data_num * (self.data_after_num+1)


class DON_Dataset(Dataset):
    ''' Dataset for loading and preprocessing the DON or DON_Conv data '''
    def __init__(self, args):
        # Read inputs
        self.latent_dim = args.latent_dim
        self.data_all = [0] + args.data_after
        self.data_after_num = args.data_after_num
        self.data_num = args.data_num
        self.features = args.features
        self.data_type = args.data_type
        self.features = args.features
        self.data_select = args.data_select
        lams = np.load(args.lam_path)
        data = np.load(args.org_path)
        orgs = data[:self.data_num, self.data_all].astype(np.float64).copy()
        orgs = orgs[:, :, self.data_select] 

        mesh = np.load(args.mesh_path)
        self.y_mesh, self.x_mesh = mesh[0], mesh[1]

        # Data Standard
        self.scaler = [[],[]]
        for i in range(2):
            for j in range(self.features):
                self.scaler[i].append(get_scaler(args, i, j))
        
        #lams (num, all, features, lam) 
        lams = lams.reshape(self.data_num * (self.data_after_num+1) , self.features, self.latent_dim) #lams (num * all, features, lam) 
        for i in range(self.features):
            lams[:,i] = self.scaler[1][i].transform(lams[:,i])
        self.lams = lams.reshape(self.data_num, self.data_after_num+1 , self.features, self.latent_dim)

        #orgs (num, all, features, H, W)
        self.orgs = orgs
        del mesh
        del lams, orgs


    def __getitem__(self, index):
        # Convert Data into PyTorch tensors
        input = torch.Tensor(self.lams[index,:1]) # (num, 1, features, lam)
        label = torch.Tensor(self.lams[index,1:])# (num, data_after, features, lam)
        return input, label

    def __len__(self):
        # Returns the size of the dataset
        return self.data_num


class lam_Dataset(Dataset):
    ''' Dataset for loading and preprocessing the lam data '''
    def __init__(self, inputs, labels, scaler):
        #input (num, after, features, lam)
        #label (num, after, features, H, W)
        self.lam = inputs.shape[3]
        self.scaler = scaler
        self.num, self.after, self.features, self.H, self.W = labels.shape
        
        inputs = inputs.reshape(-1, self.features, self.lam)
        for i in range(self.features):
            inputs[:,i] = self.scaler[1][i].inverse_transform(inputs[:,i])

        self.inputs = inputs.reshape(-1, self.features, self.lam) #input (num * after, features, lam)

        labels = labels.reshape(-1, self.features, self.H * self.W)
        for i in range(self.features):
            labels[:,i] = self.scaler[0][i].transform(labels[:,i])
        
        self.labels = labels.reshape(-1, 1, self.features, self.H, self.W) #label (num * after, 1, self.features, H, W)


    def __getitem__(self, index):
        # Convert Data into PyTorch tensors
        input = torch.Tensor(self.inputs[index])
        label = torch.Tensor(self.labels[index])
        return input, label


    def __len__(self):
        # Returns the size of the dataset
        return self.num*self.after

