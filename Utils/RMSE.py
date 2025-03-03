import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 将后处理封装成函数，同时保存各类数据
def RMSE_Vaild(predictions, ture_y, data_path, args):
    # 保存文件夹名称
    predictions_name = os.path.join(data_path, args.pred_path)
    real_name = os.path.join(data_path, args.real_path)
    image_name = os.path.join(data_path, 'image')
    os.makedirs(image_name, exist_ok=True)
    # 导入预报和真实值
    all_predictions_array = predictions
    all_test_y_array = ture_y
    try:
        if not all_predictions_array.shape == all_test_y_array.shape:
            raise ValueError('预测数据和真实数据形状不一致，请检查！')
    except ValueError as e:
        print('predictions:', all_predictions_array.shape, 'test_y:', all_test_y_array.shape)
        print(e)

    # 将 NumPy 数组转换为 pandas DataFrame
    predictions_df = pd.DataFrame(all_predictions_array, columns=[f'Pred_{(i + 1)}' for i in range(all_predictions_array.shape[1])])
    test_y_df = pd.DataFrame(all_test_y_array, columns=[f'Real_{(i + 1)}' for i in range(all_test_y_array.shape[1])])
    # 将 DataFrame 保存到 CSV 文件中
    predictions_df.to_csv(str(predictions_name), index=False)
    test_y_df.to_csv(str(real_name), index=False)

    significant_wave_height = args.significant_wave_height  # 纵摇有义值
    # 计算均方根误差 (RMSE)
    # 计算每一列的RMSE
    rmse_per_column = []
    for col_pred, col_real in zip(predictions_df.columns, test_y_df.columns):
        rmse = np.sqrt(np.mean((predictions_df[col_pred] - test_y_df[col_real]) ** 2)) / significant_wave_height
        rmse_per_column.append(rmse)

    # 创建一个包含RMSE的DataFrame
    rmse_df = pd.DataFrame({
        'Predicted_Column': predictions_df.columns,
        'RMSE': rmse_per_column
    })

    # 保存RMSE到CSV文件
    rmse_csv_filename = os.path.join(data_path, args.rmse_path)
    rmse_df.to_csv(str(rmse_csv_filename), index=False)
    # 打印结果函数
    if args.printRMSE:
        print(rmse_df)
    if args.plotRMSE:
        # 绘制误差结果图
        plt.figure(figsize=(10, 6))
        plt.plot(rmse_df['Predicted_Column'], rmse_df['RMSE'], marker='o', linestyle='-', color='b')
        plt.xlabel('Predicted Column')
        plt.ylabel('RMSE')
        plt.title('RMSE')
        plt.savefig(os.path.join(data_path, 'RMSE.png'))
        plt.close()

    def plt_prediction(test_y_df, predictions_df, data_name, look_ahead):
        # 绘制最后一个时间步的时历对比图
        plt.figure(figsize=(10, 6))
        plt.plot(test_y_df.index, test_y_df[f'Real_{look_ahead}'], label='Real', alpha=0.7)
        plt.plot(predictions_df.index, predictions_df[f'Pred_{look_ahead}'], label='Predicted', alpha=0.7)
        plt.xlabel(f'Time_step{look_ahead}')
        plt.ylabel('Value')
        plt.title(f'Time Step {look_ahead} Prediction vs Real')
        plt.legend()
        plt.grid(True)
        # 保存图表
        plt.savefig(os.path.join(data_name, f'{look_ahead}_Calendar_comparison.png'))
        plt.close()  # 关闭图表
    if args.plotCalendarComparison:
        for look_ahead_i in np.arange(10, args.look_ahead + 1, 10):
            try:
                plt_prediction(test_y_df, predictions_df, image_name, look_ahead_i)
            except Exception as e:
                pass