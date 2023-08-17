import os
import numpy as np
import argparse


class Options:
    def __init__(self, isTrain):
        self.dataset = 'MO'     # dataset: LC: Lung Cancer, MO: MultiOrgan
        self.isTrain = isTrain  # train or test mode
        self.rootDir = '' # rootdir of the data and experiment

        # --- model hyper-parameters --- #
        self.model = dict()
        self.model['name'] = 'ResUNet34'
        self.model['pretrained'] = False
        self.model['fix_params'] = False
        self.model['in_c'] = 1  # input channel

        # --- training params --- #
        self.train = dict()
        self.train['data_dir'] = '{:s}/Data/{:s}'.format(self.rootDir, self.dataset)  # path to data
        self.train['save_dir'] = '{:s}/Exp/{:s}'.format(self.rootDir, self.dataset)  # path to save results
        self.train['input_size'] = 224          # input size of the image
        self.train['train_epochs'] = 100         # number of training epochs
        self.train['batch_size'] = 8            # batch size
        self.train['checkpoint_freq'] = 20      # epoch to save checkpoints
        self.train['lr'] = 1e-3              # initial learning rate
        self.train['weight_decay'] = 5e-4       # weight decay
        self.train['log_interval'] = 37         # iterations to print training results
        self.train['ema_interval'] = 5         # eps to save ema
        self.train['workers'] = 16               # number of workers to load images
        self.train['gpus'] = [0, ]              # select gpu devices
        # --- resume training --- #
        self.train['start_epoch'] = 0    # start epoch
        self.train['checkpoint'] = ''

        # --- data transform --- #
        self.transform = dict()

        # --- test parameters --- #
        self.test = dict()
        self.test['test_epoch'] = 80
        self.test['gpus'] = [0, ]
        self.test['img_dir'] = '{:s}/Data/{:s}/images/test'.format(self.rootDir, self.dataset)
        self.test['imgh_dir'] = '{:s}/Data/{:s}/images/testh'.format(self.rootDir, self.dataset)
        self.test['label_dir'] = '{:s}/Data/{:s}/labels_instance'.format(self.rootDir, self.dataset)
        self.test['save_flag'] = True
        self.test['patch_size'] = 224
        self.test['overlap'] = 80
        self.test['save_dir'] = '{:s}/Exp/{:s}/test_results'.format(self.rootDir, self.dataset)
        self.test['checkpoint_dir'] = '{:s}/Exp/{:s}/checkpoints/'.format(self.rootDir, self.dataset)
        self.test['model_path1'] = '{:s}/checkpoint1_{:d}.pth.tar'.format(self.test['checkpoint_dir'], self.test['test_epoch'])
        self.test['model_path2'] = '{:s}/checkpoint2_{:d}.pth.tar'.format(self.test['checkpoint_dir'], self.test['test_epoch'])
        # --- post processing --- #
        self.post = dict()
        self.post['min_area'] = 20  # minimum area for an object

    def parse(self):
        """ Parse the options, replace the default value if there is a new input """
        parser = argparse.ArgumentParser(description='')
        if self.isTrain:
            parser.add_argument('--batch-size', type=int, default=self.train['batch_size'], help='input batch size for training')
            parser.add_argument('--epochs', type=int, default=self.train['train_epochs'], help='number of epochs to train')
            parser.add_argument('--lr', type=float, default=self.train['lr'], help='learning rate')
            parser.add_argument('--log-interval', type=int, default=self.train['log_interval'], help='how many batches to wait before logging training status')
            parser.add_argument('--gpus', type=int, nargs='+', default=self.train['gpus'], help='GPUs for training')
            parser.add_argument('--data-dir', type=str, default=self.train['data_dir'], help='directory of training data')
            parser.add_argument('--save-dir', type=str, default=self.train['save_dir'], help='directory to save training results')
            args = parser.parse_args()

            self.train['batch_size'] = args.batch_size
            self.train['train_epochs'] = args.epochs
            self.train['lr'] = args.lr
            self.train['log_interval'] = args.log_interval
            self.train['gpus'] = args.gpus
            self.train['data_dir'] = args.data_dir
            self.train['img_dir'] = '{:s}/images'.format(self.train['data_dir'])
            self.train['label_vor_dir'] = '{:s}/labels_voronoi'.format(self.train['data_dir'])
            self.train['label_cluster_dir'] = '{:s}/labels_cluster'.format(self.train['data_dir'])


            self.train['save_dir'] = args.save_dir
            if not os.path.exists(self.train['save_dir']):
                os.makedirs(self.train['save_dir'], exist_ok=True)

            # define data transforms for training
            self.transform['train'] = {
                'random_resize': [0.8, 1.25],
                'horizontal_flip': True,
                'vertical_flip': True,
                'random_affine': 0.3,
                'random_rotation': 90,
                'random_crop': self.train['input_size'],
                'label_encoding': 2,
                'to_tensor': 3
            }

            self.transform['val'] = {
                'random_crop': self.train['input_size'],
                'label_encoding': 2,
                'to_tensor': 2
            }

            self.transform['test'] = {
                'to_tensor': 1
            }

            self.transform['val_lin'] = {
                'to_tensor': 3
            }

        else:
            parser.add_argument('--save-flag', type=bool, default=self.test['save_flag'], help='flag to save the network outputs and predictions')
            parser.add_argument('--img-dir', type=str, default=self.test['img_dir'], help='directory of test images')
            parser.add_argument('--imgh-dir', type=str, default=self.test['imgh_dir'], help='directory of test images')
            parser.add_argument('--label-dir', type=str, default=self.test['label_dir'], help='directory of labels')
            parser.add_argument('--save-dir', type=str, default=self.test['save_dir'], help='directory to save test results')
            parser.add_argument('--gpus', type=int, nargs='+', default=self.train['gpus'], help='GPUs for training')
            parser.add_argument('--model-path1', type=str, default=self.test['model_path1'], help='train model to be evaluated')
            parser.add_argument('--model-path2', type=str, default=self.test['model_path2'],help='train model to be evaluated')
            args = parser.parse_args()
            self.test['gpus'] = args.gpus
            self.test['save_flag'] = args.save_flag
            self.test['img_dir'] = args.img_dir
            self.test['imgh_dir'] = args.imgh_dir
            self.test['label_dir'] = args.label_dir
            self.test['save_dir'] = args.save_dir
            self.test['model_path1'] = args.model_path1
            self.test['model_path2'] = args.model_path2

            if not os.path.exists(self.test['save_dir']):
                os.makedirs(self.test['save_dir'], exist_ok=True)

            self.transform['test'] = {
                'to_tensor': 1
            }

    def save_options(self):
        if self.isTrain:
            filename = '{:s}/train_options.txt'.format(self.train['save_dir'])
        else:
            filename = '{:s}/test_options.txt'.format(self.test['save_dir'])
        file = open(filename, 'w')
        groups = ['model', 'train', 'transform'] if self.isTrain else ['model', 'test', 'post', 'transform']

        file.write("# ---------- Options ---------- #")
        file.write('\ndataset: {:s}\n'.format(self.dataset))
        file.write('isTrain: {}\n'.format(self.isTrain))
        for group, options in self.__dict__.items():
            if group not in groups:
                continue
            file.write('\n\n-------- {:s} --------\n'.format(group))
            if group == 'transform':
                for name, val in options.items():
                    if (self.isTrain and name != 'test') or (not self.isTrain and name == 'test'):
                        file.write("{:s}:\n".format(name))
                        for t_name, t_val in val.items():
                            file.write("\t{:s}: {:s}\n".format(t_name, repr(t_val)))
            else:
                for name, val in options.items():
                    file.write("{:s} = {:s}\n".format(name, repr(val)))
        file.close()




