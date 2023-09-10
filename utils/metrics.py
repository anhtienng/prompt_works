from datasets.dataset import get_caption
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, cohen_kappa_score

def calculate_metrics(dataset, ground_truth_list, prediction_list):
    metrics = {}
    if dataset in ['colon-1', 'colon-2', 'prostate-1', 'prostate-2', 'gastric']:
        cancer_ground_truth_list = []
        cancer_prediction_list = []
        for i in range(len(ground_truth_list)):
            if 'benign' not in ground_truth_list[i]:
                cancer_ground_truth_list.append(ground_truth_list[i])
                cancer_prediction_list.append(prediction_list[i])
        metrics['valid_acc'] = accuracy_score(ground_truth_list, prediction_list)
        metrics['valid_cancer_acc'] = accuracy_score(cancer_ground_truth_list, cancer_prediction_list)
        metrics['valid_f1'] = f1_score(ground_truth_list, prediction_list, average='macro', labels=get_caption(dataset))
        metrics['valid_kappa'] = cohen_kappa_score(ground_truth_list, prediction_list, labels=get_caption(dataset), weights='quadratic')
        try:
            metrics['valid_avg'] = (metrics['valid_acc']+metrics['valid_cancer_acc']+metrics['valid_f1']+metrics['valid_kappa'])/4
        except:
            metrics['valid_avg'] = 0

    elif dataset in ['k19', 'k16']:
        metrics['valid_acc'] = accuracy_score(ground_truth_list, prediction_list)
        metrics['valid_f1'] = f1_score(ground_truth_list, prediction_list, average='macro', labels=get_caption(dataset))
        metrics['valid_pre'] = precision_score(ground_truth_list, prediction_list, labels=get_caption(dataset), average='macro')
        metrics['valid_rec'] = recall_score(ground_truth_list, prediction_list, labels=get_caption(dataset), average='macro')
        try:
            metrics['valid_avg'] = (metrics['valid_acc']+metrics['valid_pre']+metrics['valid_f1']+metrics['valid_rec'])/4
        except:
            metrics['valid_avg'] = 0
    else:
        raise ValueError(f'Not support dataset {dataset}')
    return metrics