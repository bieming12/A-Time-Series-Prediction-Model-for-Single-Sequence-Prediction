import time
import os
import logging
from datetime import datetime

# 配置日志记录器
def setup_logger(log_path):
    # 创建一个日志记录器
    logger = logging.getLogger('experiment_logger')
    logger.setLevel(logging.INFO)
    # 创建一个文件处理器，并设置级别为INFO
    log_file = datetime.now().strftime(os.path.join(log_path, "experiment_%Y%m%d_%H%M%S.log"))
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    # 创建一个日志格式器
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    # 将格式器添加到处理器
    file_handler.setFormatter(formatter)
    # 将处理器添加到日志记录器
    logger.addHandler(file_handler)
    return logger

def create_directories(args):
    # 添加时间戳
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    if args.times_tamp == True:
        ResultName = os.path.join('Result', f'{args.model_select}_{timestamp}')  # 进行调试时，这里是结果文件保存路径，可以根据调整的超参数修改文件名称
    else:
        ResultName = os.path.join('Result', f'{args.model_select}')  # 没有时间戳调试起来要好改的多
    print('数据文件保存地址：', ResultName)
    # 创建模型保存路径
    ResultModelPath = os.path.join(ResultName, 'Models')
    # 创建数据存档文件夹
    ResultDataPath = os.path.join(ResultName, 'DataSet')
    # 创建这两个文件夹
    os.makedirs(ResultName, exist_ok=True)
    os.makedirs(ResultDataPath, exist_ok=True)
    os.makedirs(ResultModelPath, exist_ok=True)
    # 设置日志记录器
    logger = setup_logger(ResultName)
    # 将超参数写入日志文件
    logger.info(f"Experiment started with the following hyperparameters:")
    logger.info(f"look_back: {args.look_back}")
    logger.info(f"look_ahead: {args.look_ahead}")
    logger.info(f"Data_path: {args.Data_path}")
    logger.info(f"motion_axis: {args.motion_axis}")
    logger.info(f"significant_wave_height: {args.significant_wave_height}")
    logger.info(f"model_select: {args.model_select}")
    logger.info(f'run_time: {timestamp}')
    return ResultName, ResultDataPath, ResultModelPath