import os
import pandas as pd
import numpy as np
from types import SimpleNamespace

def sliding_window(data, look_back, look_ahead):  # data数据类型是numpy
    data_x = []
    data_y = []
    for i in range(len(data) - look_back - look_ahead + 1):
        data_x.append(data[i:i +look_back])
        data_y.append(data[i + look_back:i + look_back + look_ahead])
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    data_x = np.reshape(data_x, (data_x.shape[0], data_x.shape[1], 1))
    return data_x, data_y

def Data_processing(args):
    file_path = os.path.join('data', args.Data_path)
    motion_axis = 1
    if file_path[-4:] == '.csv':
        data = pd.read_csv(file_path, usecols=[args.motion_axis])
    elif file_path[-5:] == '.xlsx':
        data = pd.read_excel(file_path, usecols=args.motion_axis)
    else:
        raise ValueError("只能接受.csv和.xlsx的文件。")
    train_data = data[:int(len(data) * args.train_scale)]
    vaild_data = data[int(len(data) * args.train_scale):int(len(data) * (args.train_scale+args.vaild_scale))]
    test_data = data[int(len(data) * (args.train_scale+args.vaild_scale)):]
    trainx, trainy = sliding_window(train_data, args.look_back, args.look_ahead)
    vaildx, vaildy = sliding_window(vaild_data, args.look_back, args.look_ahead)
    testx, testy = sliding_window(test_data, args.look_back, args.look_ahead)
    print("TrainX shape:", trainx.shape, "TrainY shape:", trainy.shape)
    print("VaildX shape:", vaildx.shape, "VaildY shape:", vaildy.shape)
    print("TestX shape:", testx.shape, "TestY shape:", testy.shape)
    trainx = trainx.reshape(-1, 1, args.look_back)
    vaildx = vaildx.reshape(-1, 1, args.look_back)
    testx = testx.reshape(-1, 1, args.look_back)
    dataset = SimpleNamespace(trainx=trainx, trainy=trainy.squeeze(), vaildx=vaildx, vaildy=vaildy.squeeze(), testx=testx, testy=testy.squeeze())
    return dataset