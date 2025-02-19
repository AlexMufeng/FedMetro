import os
import numpy as np
import pandas as pd
import pickle

def load_st_dataset(dataset):
    print(f"-_+_+_+_+_+_+_+_+_+_+_+_+_+{dataset}+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+")
    #output B, N, D
    if 'BJMetro_in10' in dataset:
        data = pd.read_csv('./data/BJMetro/inflowdata/in_10min.csv', header=None)
        data = data.to_numpy(dtype=np.float64).T # (2700, 276)
    elif 'BJMetro_in15' in dataset:
        data = pd.read_csv('./data/BJMetro/inflowdata/in_15min.csv', header=None)
        data = data.to_numpy(dtype=np.float64).T # (2700, 276)
    elif 'HZMetro_in15' in dataset:
        data = pd.read_csv('./data/HZMetro/in_15min.csv', header=None)
        data = data.to_numpy(dtype=np.float64).T #
        # data = data.drop(columns='时间').to_numpy(dtype=np.float64)
    elif 'SHMetro_in15' in dataset:
        data = pd.read_csv('./data/SHMetro/in_15min.csv')
        data = data.drop(columns='时间').to_numpy(dtype=np.float64)
    elif 'BJMetro_out15' in dataset:
        data = pd.read_csv('./data/BJMetro/outflowdata/out_15min.csv', header=None)
        data = data.to_numpy(dtype=np.float64).T # (2700, 276)
    elif 'HZMetro_out15' in dataset:
        data = pd.read_csv('./data/HZMetro/out_15min.csv', header=None)
        data = data.to_numpy(dtype=np.float64).T
        # data = data.drop(columns='时间').to_numpy(dtype=np.float64)
    elif 'SHMetro_out15' in dataset:
        data = pd.read_csv('./data/SHMetro/out_15min.csv')
        data = data.drop(columns='时间').to_numpy(dtype=np.float64)
    elif 'PeMSD4FLOW' in dataset:
        data = np.load('./data/PeMSD4/pems04.npz')
        data = data['data']
        data = data[:,:,0]
    elif 'PeMSD4OCCUPANCY' in dataset:
        data = np.load('./data/PeMSD4/pems04.npz')
        data = data['data']
        data = data[:,:,1]
    elif 'PeMSD4SPEED' in dataset:
        data = np.load('./data/PeMSD4/pems04.npz')
        data = data['data']
        data = data[:,:,2]
    elif 'PeMSD7' in dataset:
        df = pd.read_csv('./data/PeMSD7/data.csv')
        data = df.drop(columns='time').to_numpy(dtype=np.float64)
        # data = data[:2000,:]
    elif 'PeMSD8FLOW' in dataset:
        data = np.load('./data/PeMSD8/pems08.npz')
        data = data['data']
        data = data[:,:,0]
    elif 'PeMSD8OCCUPANCY' in dataset:
        data = np.load('./data/PeMSD8/pems08.npz')
        data = data['data']
        data = data[:,:,1]
    elif 'PeMSD8SPEED' in dataset:
        data = np.load('./data/PeMSD8/pems08.npz')
        data = data['data']
        data = data[:,:,2]
    elif 'METR_LA' in dataset:
        # df = pd.read_hdf(f"./data/METR_LA/metr-la.h5")
        # data = df.to_numpy()
        data = np.load(f"./data/METR_LA/metr-la.npy")
    elif 'PEMS_BAY' in dataset:
        # df = pd.read_hdf(f"./data/PEMS_BAY/pems-bay.h5")
        # data = df.to_numpy()
        data = np.load(f"./data/PEMS_BAY/pems-bay.npy")
    elif 'TAXI_1905_6min' in dataset:
        total_data = []
        for com in ['QH', 'JYJ', 'YTAX', 'ZHTC', 'JKSX']:
            data = []
            for i in range(1, 15):
                path = './data/TAXI_1905_6min/TAXI_{}/201306{:02d}.csv'.format(com, i)
                data.append(pd.read_csv(path).to_numpy())
            data = np.concatenate(data, axis=0)
            total_data.append(data)
        data = sum(total_data)
        data = data[:1000, :]
    elif 'COVID_CA' in dataset: # (335, 55)
        data = np.load("./data/COVID/CA/CA_COVID.npz")
        data = data['arr_0']
    elif 'COVID_TX' in dataset: # (335, 251)
        data = np.load("./data/COVID/TX/TX_COVID.npz")
        data = data['arr_0'][:,:250]
    elif dataset == 'COVID_CA_CTR':
        data = np.load("./data/COVID/CA/CA_COVID.npz")
        data = data['arr_0']
    elif 'COVID_CA_SGL' in dataset:
        c_id = dataset.split('_')[-1]
        with open(f'./data/COVID/CA/clients/{c_id}.pkl', 'rb') as f:
            data = pickle.load(f)
    elif 'COVID_CA_FED' in dataset:
        c_id = dataset.split('_')[-1]
        with open(f'./data/COVID/10p/CA/random_v1/client_{c_id}.pkl', 'rb') as f:
            data = pickle.load(f)
    elif dataset == 'COVID_TX_ALL':
        data = np.load("./data/COVID/TX_COVID.npz")
        data = data['arr_0']
    elif 'Blockchain_Golem' in dataset:
        data = np.load("./data/Blockchain/Golem/Golem_node_features.npz")
        data = data['arr_0']
        data = data.squeeze()
    elif 'Blockchain_Decentraland' in dataset:
        data = np.load("./data/Blockchain/Decentraland/Decentraland_node_features.npz")
        data = data['arr_0']
        data = data.squeeze()
    elif 'Blockchain_Golem_SGL' in dataset:
        c_id = dataset.split('_')[-1]
        with open(f'./data/Blockchain/Golem/clients/{c_id}.pkl', 'rb') as f:
            data = pickle.load(f)
    elif dataset == 'Blockchain_Golem_CTR':
        data = np.load("./data/Blockchain/Golem/Golem_node_features.npz")
        data = data['arr_0']
        data = data.squeeze()
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data.astype(np.float32)
