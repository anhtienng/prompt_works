import json, os

def save_config(args):
    config = {}
    for key, item in args._get_kwargs():
        config[key] = item
    out_path = os.path.join(args.out_dir, f"{args.prefix_outdir}.json")
    with open(out_path, 'w') as outfile:
        json.dump(config, outfile)


def save_info(args, log_info, writer, epoch):
    writer.add_scalar('Train/Loss', log_info['train_loss'], epoch)
    writer.add_scalar('Train/lr', log_info['lr'], epoch)

    writer.add_scalars('Valid/Metrics', log_info['val_metrics'], epoch)

    wrongs = []
    for i in range(len(log_info['ground_truth_list'])):
        if log_info['ground_truth_list'][i] != log_info['prediction_list'][i]:
            wrongs.append(
                {
                    'gt': log_info['ground_truth_list'][i],
                    'pred': log_info['prediction_list'][i]
                }
            )
    out_path = os.path.join(args.out_dir, f"wrongs_{epoch}.json")
    with open(out_path, 'w') as outfile:
        json.dump(wrongs, outfile)

def save_config_and_metric(args, best_metrics, best_epoch, run_type='valid'):
    args_str = []
    for attribute in ['epochs','bs', 'optimizer_type', 'lr', 'betas', 'encoder_type', 'encoder_prompt_len',\
                       'encoder_skip_layers','layers_dim','proj_activation','decoder_type','visual_feature_position',\
                       'decoder_prompt_len','decoder_skip_layers','prefix_outdir']:
        args_str.append(f'"{getattr(args,attribute)}"')
    args_str = ",".join(args_str) + ','

    metrics_str = []
    if args.dataset in ['colon-1', 'colon-2', 'prostate-1', 'prostate-2', 'gastric']:
        metrics =  ['valid_acc','valid_cancer_acc','valid_f1','valid_kappa']
    elif args.dataset in ['k19', 'k16']:
        metrics =  ['valid_acc','valid_f1','valid_pre','valid_rec']
    for metric in metrics:
        metrics_str.append(f'"{best_metrics[metric]}"')
    metrics_str = ",".join(metrics_str)

    out_path = os.path.join('/data4/anhnguyen/experiments/prompt_work', f"{args.dataset}-{run_type}.csv")
    with open(out_path,'a') as f:
        if best_epoch is not None:
            store_str = args_str + metrics_str + "," + str(best_epoch)
        else:
            store_str = args_str + metrics_str + "," + args.model_pth
        f.write(store_str)
        f.write("\n")
    return store_str
