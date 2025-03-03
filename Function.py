import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from data.Data import Data_processing
from Utils.logger import create_directories
from Utils.ModelSelect import create_model
from Utils.TrainModel import train_model
from Utils.TestModel import test_model
from Utils.RMSE import RMSE_Vaild

def main(args):
    # 创建文件保存路径
    Result_name, data_path, model_path = create_directories(args)  # 调试超参数时，记得修改结果保存文件名称
    # 数据处理
    dataset = Data_processing(args)
    # 模型实例化
    model = create_model(args).to(device)
    # 训练模型
    train_model(model, dataset, args=args, device=device, model_dir=model_path)
    # 测试集测试
    predictions = test_model(model_path, dataset, args=args, device=device, model_select=args.model_select)
    # 结果后处理
    RMSE_Vaild(predictions, dataset.testy, data_path, args)