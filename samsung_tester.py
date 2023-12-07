import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import random
import imgaug as ia
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import Timer
from ignite.metrics import RunningAverage
from misc.log_function import *
from misc import check_log_dir
import numpy as np
import wandb
from network_architecture import MRODNet_aspp
from datafunction import dataset
from configure.config import CONFIGURE
from loss.odrinal_regression_loss import OrdinalRegressionLoss
####
torch.autograd.set_detect_anomaly(True)


class Trainer(CONFIGURE):
    def __init__(self, _args=None):
        super(Trainer, self).__init__(_args=_args)

        self.k_ordinal_class = self.ordinal_class

        self.modelname = f"Samsung_Epoch{self.nr_epochs}_Batch" \
                         f"{self.train_batch_size}_LR{self.lr}_Seed{self.seed}_{self.loss}" \
                         f"sigmoid_depthmaprange01_Uniform"

        self.log_dir = f'/lee/project_samsung2/log/' + self.modelname
        self.result_dir = f'/lee/project_samsung2/result/' + self.modelname
        print(self.modelname)

    def check_manual_seed(self, seed):
        """
        If manual seed is not specified, choose a random one and notify it to the user
        """
        seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        ia.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        print('Using manual seed: {seed}'.format(seed=seed))
        return

    def test_step(self, net, batch, device):
        net.eval()  # infer mode

        sem_cpu, path, minmax = batch
        sem_cpu = sem_cpu.permute(0, 3, 1, 2)  # to NCHW
        # push data to GPUs
        imgs_sem = sem_cpu.to(device).float()
        # -----------------------------------------------------------
        with torch.no_grad():  # dont compute gradient
            # # assign output
            logit_depth, logit_ord = net(imgs_sem)  # a list contains all the out put of the network
            # -----------------------------------------------------------
            return dict(
                        paths=np.expand_dims(np.array(path), 1),
                        features=logit_depth.cpu().detach().numpy(),
                        minmax=np.concatenate([np.expand_dims(minmax[0], 1), np.expand_dims(minmax[1], 1)], axis=1)
                        )

    def tester(self, score_mode, commit=False):
        device = 'cuda'

        self.check_manual_seed(self.seed)
        self.best_model_dir, self.best_step = get_best_model(logdir=self.log_dir, score_mode=score_mode)
        #
        log_info_dict = {
            'commit': commit,
            'normalization_range': [0, 1],
            'batch_size': self.infer_batch_size,
            'result_dir': self.result_dir
        }

        # Define your network here
        net_parameters = {'n_channels': 1,
                          'reduction': 0.5,
                          'k_ordinal_class': 17,
                          'device': device,
                          'model_name': "efficientnet-b0",
                          'scale_factor': (4, 3),
                          'output_padding': (1, 0)}

        net = MRODNet_aspp.DrnnSddLossAspp(net_parameters, self.best_model_dir)
        #net = MRODNet_aspp.DrnnSddLossAspp(net_parameters)
        net = torch.nn.DataParallel(net).to(device)

        # --------------------------- Dataloader
        train_pairs, valid_pairs, test_pairs = dataset.prepare_sem_data()

        train_dataset = dataset.DatasetSerialSEM(train_pairs[:], mode='test')
        valid_dataset = dataset.DatasetSerialSEM(valid_pairs[:], mode='test')
        test_dataset = dataset.DatasetSerialSEM(test_pairs[:], mode='test')

        train_loader = data.DataLoader(train_dataset,
                                       num_workers=self.nr_procs_valid,
                                       batch_size=self.infer_batch_size,
                                       shuffle=False, drop_last=False)

        valid_loader = data.DataLoader(valid_dataset,
                                       num_workers=self.nr_procs_valid,
                                       batch_size=self.infer_batch_size,
                                       shuffle=False, drop_last=False)
        test_loader = data.DataLoader(test_dataset,
                                      num_workers=self.nr_procs_valid,
                                      batch_size=self.infer_batch_size,
                                      shuffle=False, drop_last=False)


        # --------------------------- Training Sequence
        trainer = Engine(lambda engine, batch: self.test_step(net, batch, device))
        valider = Engine(lambda engine, batch: self.test_step(net, batch, device))
        tester = Engine(lambda engine, batch: self.test_step(net, batch, device))

        # TODO: refactor this
        # attach running average metrics computation
        # decay of EMA to 0.95 to match tensorpack default
        timer = Timer(average=True)
        timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                     pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)
        timer.attach(valider, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                     pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)
        timer.attach(tester, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                     pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

        # attach progress bar
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer)
        pbar.attach(valider)
        pbar.attach(tester)

        trainer.accumulator = {metric: [] for metric in ['paths', 'features', 'minmax']}
        valider.accumulator = {metric: [] for metric in ['paths', 'features', 'minmax']}
        tester.accumulator = {metric: [] for metric in ['paths', 'features', 'minmax']}

        trainer.add_event_handler(Events.EPOCH_STARTED, init_accumulator)
        trainer.add_event_handler(Events.ITERATION_COMPLETED, accumulate_outputs)
        valider.add_event_handler(Events.EPOCH_STARTED, init_accumulator)
        valider.add_event_handler(Events.ITERATION_COMPLETED, accumulate_outputs)
        tester.add_event_handler(Events.EPOCH_STARTED, init_accumulator)
        tester.add_event_handler(Events.ITERATION_COMPLETED, accumulate_outputs)

        trainer.add_event_handler(Events.EPOCH_COMPLETED, testing, 'simulation_train', log_info_dict, commit)
        valider.add_event_handler(Events.EPOCH_COMPLETED, testing, 'simulation_valid', log_info_dict, commit)
        tester.add_event_handler(Events.EPOCH_COMPLETED, testing, 'submission', log_info_dict, commit)

        # trainer.run(train_loader, 1)
        # valider.run(valid_loader, 1)
        tester.run(test_loader, 1)


        return
    ###
    def run(self):
        self.tester('loss', commit=True)
        return


####
if __name__ == '__main__':
    torch.backends.cudnn.enabled = False  # cuDNN error: CUDNN_status_mapping_error
    parser = argparse.ArgumentParser()
    # parser.add_argument('--gpu', default='0,1', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--seed', type=int, default=5, help='number')
    args = parser.parse_args()

    trainer = Trainer(_args=args)
    # if args.gpu:
    #     os.environ['CUDA_VISIBLE_DEVICES']="0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', device)
    print('Count of using GPUs:', torch.cuda.device_count())
    print('Current cuda device:', torch.cuda.current_device())
    trainer.run()