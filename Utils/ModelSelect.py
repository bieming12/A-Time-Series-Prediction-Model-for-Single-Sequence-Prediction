import torch.nn as nn
from types import SimpleNamespace
from algorithm.CommonlyUsed import LSTMModel, GRUModel, ANNModel
# 选择性的创建模型
def create_model(args):
    if args.model_select == 'ANN':
        model = ANNModel(input_size=1, sequence_length = args.look_back, output_size=args.look_ahead, num = args.look_back)
    elif args.model_select == 'GRU':
        model = GRUModel(input_size=1, output_size=args.look_ahead)
    elif args.model_select == 'LSTM':
        model = LSTMModel(input_size=1, output_size=args.look_ahead)
    else:
        raise ValueError("模型名称错误！")
    return model