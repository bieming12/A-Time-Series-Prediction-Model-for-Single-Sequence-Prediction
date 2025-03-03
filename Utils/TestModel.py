import os
import torch
import numpy as np
from Utils.ModelSelect import create_model

# 测试模型
def test_model(model_path, dataset, args, device, model_select = 'ANN'):
    best_model_path = os.path.join(model_path, 'best_model.pth')
    model = create_model(args)
    model.load_state_dict(torch.load(best_model_path, weights_only=False))
    model.to(device)
    model.eval()
    # 准备测试数据
    test_X_tensor = torch.tensor(dataset.testx).float()
    with torch.no_grad():
        for test_X_batch in np.arange(0, dataset.testx.shape[0], args.batch_size):
            # 这一步操作以实现对于不完整批次的正常预测
            if test_X_batch + args.batch_size < dataset.testx.shape[0]:
                test_X_batch_tensor = test_X_tensor[test_X_batch:test_X_batch + args.batch_size]
            else:
                test_X_batch_tensor = test_X_tensor[test_X_batch:]
            test_X_batch_tensor = test_X_batch_tensor.to(device)
            model_output = model(test_X_batch_tensor)
            model_output = model_output.squeeze()
            predictions = model_output.cpu().numpy()
            if test_X_batch == 0:
                all_predictions = predictions
            else:
                try:
                    all_predictions = np.concatenate((all_predictions, predictions), axis=0)
                except ValueError:
                    print(f'测试样本的最后一个数组维度为{predictions.ndim},建议调整小批量大小，防止最后一个数组维度缺1')
    return all_predictions