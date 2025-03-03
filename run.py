import argparse
from Function import main
# 可修改变量
parser = argparse.ArgumentParser(description='简易时序预测函数')

# 基本超参数
parser.add_argument('--look_back', type=int, default=100, help='输入时间序列，即回看时间步')
parser.add_argument('--look_ahead', type=int, default=100, help='输出时间序列，即预测步长')
parser.add_argument('--Data_path', type=str, default='data.csv',help='文件名称，请将对应的CSV文件放入项目文件的data文件夹中')
parser.add_argument('--motion_axis', type=str, default='values', help='输入文件的列名')
parser.add_argument('--train_scale', type=float, default=0.8, help='训练数据集所占的比例')
parser.add_argument('--vaild_scale', type=float, default=0.1, help='验证数据集所占的比例')
parser.add_argument('--significant_wave_height', type=float, default=1.0,help='对于多种不同数据之间对比rmse值保证标准化用')

# 模型选择
parser.add_argument('--model_select', type=str, default='LSTM', help='模型选择，可选：ANN, GRU, LSTM')
parser.add_argument('--batch_size', type=int, default=64, help='小批量大小')
parser.add_argument('--epochs', type=int, default=300, help='训练轮数')
parser.add_argument('--patience', type=int, default=20, help='早停的耐心数')

# 结果文件名称
parser.add_argument('--pred_path', type=str, default='predictions.csv', help='预报结果文件名称')
parser.add_argument('--real_path', type=str, default='real.csv', help='真实结果文件名称')
parser.add_argument('--rmse_path', type=str, default='rmse.csv', help='rmse结果文件名称')
parser.add_argument('--printRMSE', type=bool, default=True, help='是否打印误差结果')
parser.add_argument('--plotRMSE', type=bool, default=True, help='是否绘制误差图像')
parser.add_argument('--plotCalendarComparison', type=bool, default=True, help='是否绘制时历对比图')
parser.add_argument('--times_tamp', type=bool, default=False, help='是否为模型文件添加时间戳')

# 解析参数
args = parser.parse_args()

main(args)