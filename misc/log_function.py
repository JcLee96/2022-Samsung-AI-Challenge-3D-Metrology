import numpy as np
from termcolor import colored
import json
import cv2
import os
import wandb
import zipfile

def get_json_data(logdir):
    with open(logdir + '/stats.json', 'r') as f:
        json_data = json.load(f)
    return json_data

# def get_best_step(log, score_mode):
#     best_score, best_step = 1e8, 0
#     for i, step in enumerate(log):
#         current_score_1 = float(log[step]['valid-' + 'loss_rmse'])
#         current_score_2 = float(log[step]['valid-' + 'loss_ord'])
#         if current_score_1+current_score_2 < best_score:
#             best_score = current_score_1+current_score_2
#             best_step = step
#     print(f'best valid score [{score_mode}]:{best_score}')
#     print(f'best valid step [{score_mode}]:{best_step}')
#     return best_step

def get_best_step(log, score_mode):
    best_score, best_step = 1e8, 0
    for i, step in enumerate(log):
        current_score = float(log[step]['valid-' + score_mode])
        if current_score < best_score:
            best_score = current_score
            best_step = step
    print(f'best valid score [{score_mode}]:{best_score}')
    print(f'best valid step [{score_mode}]:{best_step}')
    return best_step

def get_best_model(logdir, score_mode=None):
    log = get_json_data(logdir)
    step = get_best_step(log, score_mode)

    net_dir = logdir + f'/_net_{step}.pth'

    return net_dir, step

def update_log(output, epoch, prefix, color, log_file, commit):
    max_length = len(max(output.keys(), key=len))
    output_length = (len(output) - 1)
    # print
    stat_dict ={}
    for idx, metric in enumerate(output):
        key = colored(prefix + '-' + metric.ljust(max_length) + ':', color)
        print("-----%s" % key, end=' ')
        print("%0.7f" % output[metric])
        stat_dict['%s-%s' % (prefix, metric)] = output[metric]

        if idx == output_length:
            wandb.log({prefix + '-' + metric.ljust(max_length): output[metric]}, commit=commit)
        else:
            wandb.log({prefix + '-' + metric.ljust(max_length): output[metric]}, commit=False)

    with open(log_file) as json_file:
        json_data = json.load(json_file)

    current_epoch = str(epoch)
    if current_epoch in json_data:
        old_stat_dict = json_data[current_epoch]
        stat_dict.update(old_stat_dict)
    current_epoch_dict = {current_epoch : stat_dict}
    json_data.update(current_epoch_dict)

    with open(log_file, 'w') as json_file:
        json.dump(json_data, json_file)

def training(engine, info, commit=False):
    """
    running training measurement
    """
    train_output = engine.state.metrics
    train_output['lr'] = info['optimizer'].param_groups[0]['lr']
    output = dict(loss=train_output['loss'], loss_rmse=train_output['loss_rmse'], loss_ord=train_output['loss_ord'])
    update_log(output, engine.state.epoch, 'train', 'green', info['json_file'], commit=commit)

#validation 실행 함수
def inference(engine, inferer, prefix, dataloader, info, commit):
    """
    running inference measurement
    """
    # valider.accumulator = {prob:[], true:[]}로 생성 및 Init
    #inferer.accumulator = {metric: [] for metric in info['metric_names']}
    # validation 수행. (수행시마다의 결과값이 누적되어야함.)
    inferer.run(dataloader)
    # {prob:[], true:[]}으로 이루어진 batchsize * iteration개의 결과를 이용해 acc와 confusion matrix를 생성
    metrics = inferer.state.metrics
    output = dict(loss=metrics['loss'], loss_rmse=metrics['loss_rmse'], loss_ord=metrics['loss_ord'])
    update_log(output, engine.state.epoch, prefix, 'red', info['json_file'], commit)

    # if prefix in ['test'] and engine.state.epoch == info['nr_epochs']:
    #     get_best_score(info['json_file'])

def init_accumulator(engine):
    engine.accumulator = {'paths': [], 'features': [], 'minmax': []}

#train, validation 결과 누적 함수
def accumulate_outputs(engine):
    batch_output = engine.state.output
    for key, item in batch_output.items():
        engine.accumulator[key].extend([item])
    return

def testing(engine, prefix, info, commit=False):
    def uneven_seq_to_np(seq, size):
        item_count = size * (len(seq) - 1) + len(seq[-1])
        cat_array = np.zeros((item_count,) + seq[0][0].shape, seq[0].dtype)
        # BUG: odd len even
        for idx in range(0, len(seq)-1):
            cat_array[idx * size: (idx+1) * size] = seq[idx]
        cat_array[(idx+1) * size:] = seq[-1]
        return cat_array
    features = uneven_seq_to_np(engine.accumulator['features'], info['batch_size'])
    paths = uneven_seq_to_np(engine.accumulator['paths'], info['batch_size'])
    minmax_values = uneven_seq_to_np(engine.accumulator['minmax'], info['batch_size'])
    norm_min, norm_max = info['normalization_range'][0], info['normalization_range'][1]

    os.makedirs(f"{info['result_dir']}/zips/", exist_ok=True)
    os.makedirs(f"{info['result_dir']}/{prefix}/", exist_ok=True)
    save_dir = f"{info['result_dir']}/{prefix}/"

    sub_imgs = []
    for path, pred_img, minmax in zip(paths, features, minmax_values):
        path = path[0]
        pred_img = pred_img.transpose(1, 2, 0)
        min, max = minmax[0], minmax[1]
        #img=np.clip(np.round(pred_img*255,0),0,255).astype('uint8')
        img = np.clip(np.round((pred_img-norm_min) * (max-min) / (norm_max-norm_min) + min, 0), 0, 255).astype('uint8')
        img_name = path.split('/')[-1]

        if img_name[-4:] in ['.png']:
            cv2.imwrite(save_dir+img_name, img)
            sub_imgs.append(img_name)
    submission = zipfile.ZipFile(f"{info['result_dir']}/zips/{prefix}.zip", 'w')
    os.chdir(f"{info['result_dir']}/{prefix}/")
    for path in sub_imgs:
        submission.write(path)
    submission.close()








