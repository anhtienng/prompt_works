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
