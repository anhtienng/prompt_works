import json, os
import torch
from torch.utils.data import DataLoader
from datetime import datetime

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
                       'decoder_prompt_len','decoder_skip_layers','prefix_outdir','decoder_skip_layers_for_visual']:
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

def get_optimizer(args, model):
    if args.optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=args.betas)
    elif args.optimizer_type == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=args.betas)
    else:
        raise NotImplementedError
    
    return optimizer

def get_dataloader(args, train_dataset, valid_dataset):
    train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, drop_last=True, num_workers=args.num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.bs, shuffle=True, drop_last=False, num_workers=args.num_workers)

    return train_dataloader, valid_dataloader

def process_args(args):
    log_skip_layers_encoder = str(args.encoder_skip_layers)[1:-1].replace(', ', '') if args.encoder_skip_layers != [] else 'full'
    log_skip_layers_decoder = str(args.decoder_skip_layers)[1:-1].replace(', ', '') if args.decoder_skip_layers != [] else 'full'
    log_skip_layers_decoder_visual = str(args.decoder_skip_layers_for_visual)[1:-1].replace(', ', '') if args.decoder_skip_layers_for_visual != [] else 'full'
    log_skip_layers_lora_e = str(args.encoder_lora_skip_layers)[1:-1].replace(', ', '') if args.encoder_lora_skip_layers != [] else 'full'
    log_skip_layers_lora_d = str(args.decoder_lora_skip_layers)[1:-1].replace(', ', '') if args.decoder_lora_skip_layers != [] else 'full'
    
    args.scheduler_k = args.epochs

    now = datetime.now()
    if args.prompt_type != 'lora':
        args.prefix_outdir = '-'.join((args.dataset,
                                    args.optimizer_type,
                                    args.scheduler_type,
                                    args.lora_r,
                                    args.lora_alpha,
                                    args.encoder_type,
                                    str(args.encoder_prompt_len),
                                    log_skip_layers_encoder,
                                    args.decoder_type,
                                    str(args.decoder_prompt_len),
                                    log_skip_layers_decoder,
                                    log_skip_layers_decoder_visual,
                                    args.prefix_outdir,
                                    str(now)[-3:]
                                    ))
    else:
        args.prefix_outdir = '-'.join((args.dataset,
                                    args.optimizer_type,
                                    args.scheduler_type,
                                    args.encoder_type,
                                    log_skip_layers_lora_e,
                                    args.decoder_type,
                                    log_skip_layers_lora_d,
                                    log_skip_layers_decoder_visual,
                                    args.prefix_outdir,
                                    str(now)[-3:]
                                    ))

    args.out_dir = os.path.join(args.out_dir,args.prompt_type,args.prefix_outdir)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    
    save_config(args)
    args.device = torch.device(f'cuda:{args.device}')