import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix
from tqdm import tqdm

from utils.serialization import save_as_file


def model_evaluation(global_model, test_dataloader, device, file_path, save_matrix=False):
    criterion = torch.nn.CrossEntropyLoss()
    global_model.eval()  # switch to eval status
    y_true = []
    y_predict = []
    loss_sum = []
    with torch.no_grad():
        for (batch_x, batch_y) in tqdm(test_dataloader):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            batch_y_predict = global_model(batch_x)

            loss = criterion(batch_y_predict, batch_y)
            batch_y_predict = torch.argmax(batch_y_predict, dim=1)
            y_true.append(batch_y)
            y_predict.append(batch_y_predict)
            loss_sum.append(loss.item())

    y_true = torch.cat(y_true, 0)
    y_predict = torch.cat(y_predict, 0)
    loss = sum(loss_sum) / len(loss_sum)

    y_true = y_true.cpu()
    y_predict = y_predict.cpu()

    print(classification_report(y_true, y_predict, target_names=test_dataloader.dataset.classes))

    if save_matrix:
        matrix = confusion_matrix(y_true, y_predict)
        save_as_file(matrix, f'{file_path}/confusion_matrix')

    return {
        'acc': accuracy_score(y_true, y_predict),
        'precision': precision_score(y_true, y_predict, average='micro'),
        'recall': recall_score(y_true, y_predict, average='micro'),
        'f1': f1_score(y_true, y_predict, average='micro'),
        "loss": loss,
    }

