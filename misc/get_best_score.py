import argparse
import json
import pandas as pd
import os

def get_json_data(logdir):
    with open(logdir + '/stats.json', 'r') as f:
        json_data = json.load(f)
    return json_data


def get_best_step(log, score_mode, limit=-1):
    best_score, best_step = 0., 0
    for i, step in enumerate(log):
        if i == limit:
            break
        current_score = float(log[step]['valid-' + score_mode])
        if best_score < current_score:
            best_score = current_score
            best_step = step
    print(f'best valid score :{best_score}')
    print(f'best valid step :{best_step}')
    return best_step

def convert_confmat_strType_to_df(confmat_strType):
    confmat_strType = confmat_strType.strip("[[" "]]").split("],[")
    confmat = []
    for row in confmat_strType:
        confmat.append(list(map(int, row.split(","))))
    return confmat

def create_df(score_mode, confmat, log, step):
    df = pd.DataFrame(confmat)
    df_tmp = pd.DataFrame(columns=df.columns, index=['True']).fillna('')
    df = pd.concat([df_tmp, df], axis=0)
    df['ALL'] = df.sum(axis=1)
    df = df.reset_index().rename(columns={'index': 'Pred'})
    df['ALL'][0] = ''
    df_tmp = pd.DataFrame(columns=['Score']).fillna('')
    df = pd.concat([df, df_tmp], axis=1).fillna('')
    if len(df) > 6 :
        df['Score'][len(df)-6] = f"{score_mode}-Score"
        df['Score'][len(df)-5] = f"step : {step}"
        df['Score'][len(df)-4] = f"acc : {log[f'{step}'][f'{score_mode}-acc']}"
        df['Score'][len(df)-3] = f"f1_score : {log[f'{step}'][f'{score_mode}-f1_score']}"
        df['Score'][len(df)-2] = f"precision : {log[f'{step}'][f'{score_mode}-precision']}"
        df['Score'][len(df)-1] = f"recall : {log[f'{step}'][f'{score_mode}-recall']}"
    else :
        df['Score'][len(df)-1] = f"{score_mode}-Score\nstep : {step}," \
                                 f"acc : {log[f'{step}'][f'{score_mode}-acc']}," \
                                 f"f1_score : {log[f'{step}'][f'{score_mode}-f1_score']}," \
                                 f"precision : {log[f'{step}'][f'{score_mode}-precision']}," \
                                 f"recall : {log[f'{step}'][f'{score_mode}-recall']}"
    print(step)
    print(df)
    return df

def get_best_model(logdir, score_mode=None, limit=-1):
    log = get_json_data(logdir)
    step = get_best_step(log, score_mode, limit=limit)

    net_dir = logdir + f'/_net_{step}.pth'

    return net_dir

def get_model(logdir, step):

    net_dir = logdir + f'/_net_{step}.pth'

    return net_dir

def get_best_score(logdir, log=None, score_mode=None):
    log = get_json_data(logdir)
    step = get_best_step(log, score_mode)

    try :
        confmat_strType = log[f'{step}']['train-ema-conf_mat']
        train_cf = convert_confmat_strType_to_df(confmat_strType)
        train_df = create_df('train-ema', train_cf, log, step)
    except:
        print('There is no train-ema-conf_mat')

    try :
        confmat_strType = log[f'{step}']['valid-conf_mat']
        valid_cf = convert_confmat_strType_to_df(confmat_strType)
        valid_df = create_df('valid', valid_cf, log, step)
    except:
        print('There is no valid-conf_mat')
    try:
        confmat_strType = log[f'{step}']['test-conf_mat']
        test_cf = convert_confmat_strType_to_df(confmat_strType)
        test_df = create_df('test', test_cf, log, step)
    except:
        print('There is no test-conf_mat')
        test_df = None

    df = pd.concat([train_df, valid_df], axis=0)

    if test_df is not None:
        df = pd.concat([df, test_df], axis=0)

    logdir = logdir.replace('gastric','csv/gastric')
    os.makedirs(logdir, exist_ok=True)

    if os.path.isfile(logdir+'/bestscore.xlsx'):
        os.remove(logdir+'/bestscore.xlsx')
        print(f'{logdir}/bestscore.xlsx is existed, so remove it')


    df.to_excel(logdir+'/bestscore.xlsx', index=False)
    print(f'{logdir}/bestscore.xlsx is saved')
    # writer = pd.ExcelWriter(logdir+'/bestscore.xlsx', engine='xlsxwriter')
    # df.to_excel(writer, sheet_name=df, index=False)
    # writer.save()

    return 0


if __name__ == '__main__':
    log_path = '/log/'

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='model name')
    parser.add_argument('--score_mode', type=str, default='f1_score', help='f1_score, acc')
    parser.add_argument('--mode', default=False)
    parser.add_argument('--limit', default=-1)
    args = parser.parse_args()

    logdir = log_path+args.model_name
    print(args)

    get_best_score(logdir=logdir, score_mode=args.score_mode, test_mode=args.mode, limit=args.limit)  # mode = ['f1_score', 'acc']