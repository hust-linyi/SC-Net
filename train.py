from PIL import Image
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.utils.data
import os
import shutil
import numpy as np
import logging
import random
from model.modelW import ResWNet34
from model.model_WNet import WNet
import utils.utils as utils
from dataloaders.dataset import DataFolder, DataFolderTest
from dataloaders.my_transforms import get_transforms
from options import Options
from utils.divergence import loss_CT
from utils.combine import Combine
from rich.logging import RichHandler
from rich import print
import imageio
from tqdm import tqdm
import shutil


def main():
    global opt, num_iter, logger, logger_results
    opt = Options(isTrain=True)
    opt.parse()
    opt.save_options()


    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in opt.train['gpus'])

    # set up logger
    logger, logger_results = setup_logging(opt)
    num = 1
    random.seed(num)
    os.environ['PYTHONHASHSEED'] = str(num)
    np.random.seed(num)
    torch.manual_seed(num)
    torch.cuda.manual_seed(num)
    torch.cuda.manual_seed_all(num)

    # ----- create model ----- #
    model1 = ResWNet34(seg_classes = 2, colour_classes = 3)
    model1 = nn.DataParallel(model1)
    model1 = model1.cuda()

    model2 = WNet(n_channels = 1, seg_classes = 2, colour_classes = 3)
    # model2 = ResWNet34(seg_classes = 2, colour_classes = 3)
    model2 = nn.DataParallel(model2)
    model2 = model2.cuda()
    cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # ----- define optimizer ----- #
    optimizer1 = torch.optim.Adam(model1.parameters(), opt.train['lr'], betas=(0.9, 0.99), weight_decay=opt.train['weight_decay'])
    optimizer2 = torch.optim.Adam(model2.parameters(), opt.train['lr'], betas=(0.9, 0.99), weight_decay=opt.train['weight_decay'])
    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=30, gamma=0.1)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=30, gamma=0.1)
    # ----- define criterion ----- #
    criterion = torch.nn.NLLLoss(ignore_index=2).cuda()
    criterion_H = torch.nn.L1Loss().cuda()

    # ----- load data ----- #
    data_transforms = {'train': get_transforms(opt.transform['train']),
                       'val': get_transforms(opt.transform['val']),
                       'test': get_transforms(opt.transform['val_lin'])}

    img1_dir = '{:s}/train1'.format(opt.train['img_dir'])
    img2_dir = '{:s}/train2'.format(opt.train['img_dir'])
    imgh1_dir = '{:s}/trainh1'.format(opt.train['img_dir'])
    imgh2_dir = '{:s}/trainh2'.format(opt.train['img_dir'])
    val_dir = '{:s}/val'.format(opt.train['img_dir'])
    valh_dir = '{:s}/valh'.format(opt.train['img_dir'])
    target_vor1_dir = '{:s}/train1'.format(opt.train['label_vor_dir'])
    target_vor2_dir = '{:s}/train2'.format(opt.train['label_vor_dir'])
    val_vor_dir = '{:s}/val'.format(opt.train['label_vor_dir'])
    target_cluster1_dir = '{:s}/train1'.format(opt.train['label_cluster_dir'])
    target_cluster2_dir = '{:s}/train2'.format(opt.train['label_cluster_dir'])
    val_cluster_dir = '{:s}/val'.format(opt.train['label_cluster_dir'])
    weight12_dir = os.path.join(opt.train['save_dir'], 'weight12')
    weight21_dir = os.path.join(opt.train['save_dir'], 'weight21')

    dir_list1 = [img1_dir, imgh1_dir, weight12_dir, target_vor1_dir, target_cluster1_dir]
    dir_list2 = [img2_dir, imgh2_dir, weight21_dir, target_vor2_dir, target_cluster2_dir]
    val_list = [val_dir, valh_dir, val_vor_dir, val_cluster_dir]
    post_fix = ['_h.png', '_h.png', '_label_vor.png', '_label_cluster.png']
    num_channels = [3, 1, 1, 3, 3]
    val_post_fix = ['_h.png', '_label_vor.png', '_label_cluster.png']
    val_num_channels = [3, 1, 3, 3]
    train_set1 = DataFolder(dir_list1, post_fix, num_channels, data_transforms['train'])
    train_loader1 = DataLoader(train_set1, batch_size=opt.train['batch_size'], shuffle=True,
                               num_workers=opt.train['workers'])
    train_set2 = DataFolder(dir_list2, post_fix, num_channels, data_transforms['train'])
    train_loader2 = DataLoader(train_set2, batch_size=opt.train['batch_size'], shuffle=True,
                               num_workers=opt.train['workers'])
    val_set = DataFolder(val_list, val_post_fix, val_num_channels, data_transforms['val'])
    val_loader = DataLoader(val_set, batch_size=opt.train['batch_size'], shuffle=True, num_workers=opt.train['workers'])

    # test data
    test_img_dir = opt.test['img_dir']
    test_imgh_dir = opt.test['imgh_dir']
    test_label_dir = opt.test['label_dir']
    test_post_fix = ['.png', '_label.png']
    test_num_channels = [3, 1, 1]
    test_dir = [test_img_dir, test_imgh_dir, test_label_dir]
    test_set = DataFolderTest(test_dir, test_post_fix, test_num_channels, data_transforms['test'])
    test_loader = DataLoader(test_set, batch_size=opt.train['batch_size'], shuffle=False, drop_last=False,
                               num_workers=opt.train['workers'])

    # ----- training and validation ----- #
    num_epoch = opt.train['train_epochs']
    num_iter = num_epoch * len(train_loader1)
    # print training parameters
    logger.info("=> Initial learning rate: {:g}".format(opt.train['lr']))
    logger.info("=> Batch size: {:d}".format(opt.train['batch_size']))
    logger.info("=> Number of training iterations: {:d}".format(num_iter))
    logger.info("=> Training epochs: {:d}".format(opt.train['train_epochs']))
    min_loss1 = 100
    min_loss2 = 100
    for epoch in range(opt.train['start_epoch'], num_epoch):
        # train for one epoch or len(train_loader) iterations
        logger.info('Epoch: [{:d}/{:d}]'.format(epoch+1, num_epoch))
        if epoch % opt.train['ema_interval'] == 0:
            if epoch == 0:
                if os.path.exists(os.path.join(opt.train['save_dir'], 'weight12')):
                    shutil.rmtree(os.path.join(opt.train['save_dir'], 'weight12'))
                    shutil.rmtree(os.path.join(opt.train['save_dir'], 'weight21'))
            ensemble_prediction(imgh2_dir, os.path.join(opt.train['save_dir'], 'weight21'), model1)
            ensemble_prediction(imgh1_dir, os.path.join(opt.train['save_dir'], 'weight12'), model2)

        test_dice1 = val_lin(model1, test_loader)
        test_dice2 = val_lin(model2, test_loader)
        print(f'test_dice1: {test_dice1: .4f}\ttest_dice2: {test_dice2: .4f}')

        train_results = train(train_loader1, train_loader2, model1, model2, optimizer1, optimizer2, criterion, criterion_H, epoch, num_epoch)
        train_loss, train_loss_vor, train_loss_cluster, train_loss_ct, train_loss_self = train_results
        state1 = {'epoch': epoch + 1, 'state_dict': model1.state_dict(), 'optimizer': optimizer1.state_dict()}
        state2 = {'epoch': epoch + 1, 'state_dict': model2.state_dict(), 'optimizer': optimizer2.state_dict()}
        if (epoch + 1) < (num_epoch - 4):
            cp_flag = (epoch + 1) % opt.train['checkpoint_freq'] == 0
        else:
            cp_flag = True
        save_checkpoint1(state1, epoch, opt.train['save_dir'], cp_flag)
        save_checkpoint2(state2, epoch, opt.train['save_dir'], cp_flag)

        # save the training results to txt files
        logger_results.info('{:d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
                            .format(epoch+1, train_loss, train_loss_vor, train_loss_cluster, train_loss_ct, train_loss_self, test_dice1, test_dice2))
        scheduler1.step()
        scheduler2.step()

        val_loss1 = val(val_loader, model1, criterion)
        val_loss2 = val(val_loader, model2, criterion)

        if val_loss1 < min_loss1:
            print(f'val_loss1: {val_loss1:.4f}', f'min_loss1: {min_loss1:.4f}')
            min_loss1 = val_loss1
            save_bestcheckpoint1(state1, opt.train['save_dir'])
        if val_loss2 < min_loss2:
            print(f'val_loss2: {val_loss2:.4f}', f'min_loss2: {min_loss2:.4f}')
            min_loss2 = val_loss2
            save_bestcheckpoint2(state2, opt.train['save_dir'])
    for i in list(logger.handlers):
        logger.removeHandler(i)
        i.flush()
        i.close()
    for i in list(logger_results.handlers):
        logger_results.removeHandler(i)
        i.flush()
        i.close()


def train(train_loader1, train_loader2, model1, model2, optimizer1, optimizer2, criterion, criterion_H, epoch, num_epoch):
    # list to store the average loss for this epoch
    results = utils.AverageMeter(5)
    # switch to train mode
    model1.train()
    model2.train()
    weight12, weight21 = None, None
    for i, (sample1, sample2) in enumerate(zip(train_loader1, train_loader2)):
        input1, inputh1, weight12, vor1, cluster1 = sample1
        input2, inputh2, weight21, vor2, cluster2 = sample2
        if vor1.dim() == 4:
            vor1 = vor1.squeeze(1)
        if vor2.dim() == 4:
            vor2 = vor2.squeeze(1)
        if cluster1.dim() == 4:
            cluster1 = cluster1.squeeze(1)
        if cluster2.dim() == 4:
            cluster2 = cluster2.squeeze(1)
        inputh_var1 = inputh1.float().cuda()
        inputh_var2 = inputh2.float().cuda()

        # compute output
        output11, output11l = model1(inputh_var1)
        output22, output22l = model2(inputh_var2)

        log_prob_maps1 = F.log_softmax(output11, dim=1)
        loss_vor1 = criterion(log_prob_maps1, vor1.cuda())
        loss_cluster1 = criterion(log_prob_maps1, cluster1.cuda())

        log_prob_maps2 = F.log_softmax(output22, dim=1)
        loss_vor2 = criterion(log_prob_maps2, vor2.cuda())
        loss_cluster2 = criterion(log_prob_maps2, cluster2.cuda())

        loss_vor = loss_vor1 + loss_vor2
        loss_cluster = loss_cluster1 + loss_cluster2

        pseudo12 = Combine(weight12.float().cuda(), cluster1)
        Pseudo12 = Variable(pseudo12, requires_grad=False)
        pseudo21 = Combine(weight21.float().cuda(), cluster2)
        Pseudo21 = Variable(pseudo21, requires_grad=False)

        loss_ct1 = loss_CT(output11, Pseudo12)
        loss_ct2 = loss_CT(output22, Pseudo21)
        loss_ct = loss_ct1 + loss_ct2

        loss_self1 = criterion_H(output11l, input1.cuda())
        loss_self2 = criterion_H(output22l, input2.cuda())
        loss_self = loss_self1 + loss_self2

        loss = loss_vor + loss_cluster + loss_ct * (epoch/num_epoch)**2 + loss_self * (1 - (epoch/num_epoch)**2) * 0.1

        result = [loss.item(), loss_vor.item(), loss_cluster.item(), loss_ct.item(), loss_self.item()]

        results.update(result, input1.size(0))

        # compute gradient and do SGD step
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        loss.backward()
        optimizer1.step()
        optimizer2.step()

        if i % opt.train['log_interval'] == 0:
            logger.info('Iteration: [{:d}/{:d}]'
                        '\tLoss {r[0]:.4f}'
                        '\tLoss_vor {r[1]:.4f}'
                        '\tLoss_cluster {r[2]:.4f}'
                        '\tLoss_ct {r[3]:.4f}'
                        '\tLoss_self {r[4]:.4f}'.format(i, len(train_loader1), r=results.avg))

    logger.info('===> Train Avg: Loss {r[0]:.4f}'
                '\tloss_vor {r[1]:.4f}'
                '\tloss_cluster {r[2]:.4f}'
                '\tloss_ct {r[3]:.4f}'
                '\tloss_self {r[4]:.4f}'.format(r=results.avg))

    return results.avg

def val(val_loader, model, criterion):
    model.eval()
    results = 0
    for i, sample in enumerate(val_loader):
        input, inputh, target1, target2 = sample
        if target1.dim() == 4:
            target1 = target1.squeeze(1)
        if target2.dim() == 4:
            target2 = target2.squeeze(1)

        input_var = inputh.cuda()

        # compute output
        output, _  = model(input_var)
        log_prob_maps = F.log_softmax(output, dim=1)
        loss_vor = criterion(log_prob_maps, target1.cuda())
        loss_cluster = criterion(log_prob_maps, target2.cuda())
        result = loss_vor.item() + loss_cluster.item()
        results += result
    val_loss = results / (opt.train['batch_size'] * len(val_loader))
    return val_loss


def dice_coeff(pred, gt):
    target = torch.zeros_like(gt)
    target[gt > 0.5] = 1
    target = gt
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2)
    dice = (2. * intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
    avg_dice = dice.sum() / num
    return avg_dice

def val_lin(model, test_loader):
    model.eval()
    metric_names = ['dice']
    all_result = utils.AverageMeter(len(metric_names))

    size = opt.test['patch_size']
    overlap = opt.test['overlap']

    for _, inputh, gt in test_loader:
        if gt.dim() == 4:
            gt = gt.squeeze(1).cuda()
        output = utils.split_forward(model, inputh, size, overlap)
        log_prob_maps = F.softmax(output, dim=1)
        pred_labeled = torch.argmax(log_prob_maps, axis=1)  # prediction
        dice = dice_coeff(pred_labeled, gt).cpu().numpy()
        all_result.update([dice])
    dice_avg = all_result.avg[0]
    return dice_avg


def ensemble_prediction(img_dir, save_dir, model, alpha=0.1):
    ## save pred into the experiment fold
    ## load all the training images
    img_names = os.listdir(img_dir)
    img_process = tqdm(img_names)
    test_transform = get_transforms(opt.transform['test'])
    print('[bold magenta]saving EMA weights ...[/bold magenta]')
    for img_name in img_process:
        img = Image.open(os.path.join(img_dir, img_name))
        input = test_transform((img,))[0].unsqueeze(0).cuda()
        output, _ = model(input)
        log_prob_maps = F.softmax(output, dim=1)
        pred = log_prob_maps.squeeze(0).cpu().detach().numpy()[1]

        try:
            weight = imageio.imread(os.path.join(save_dir, img_name))
            weight = np.array(weight)
            weight = alpha * pred + (1 - alpha) * weight
        except:
            weight = pred
            os.makedirs(save_dir, exist_ok=True)
        imageio.imsave(os.path.join(save_dir, img_name), (weight * 255).astype(np.uint8))


def save_checkpoint1(state, epoch, save_dir, cp_flag):
    cp_dir = '{:s}/checkpoints'.format(save_dir)
    if not os.path.exists(cp_dir):
        os.mkdir(cp_dir)
    filename = '{:s}/checkpoint1.pth.tar'.format(cp_dir)
    torch.save(state, filename)
    if cp_flag:
        shutil.copyfile(filename, '{:s}/checkpoint1_{:d}.pth.tar'.format(cp_dir, epoch+1))
def save_checkpoint2(state, epoch, save_dir, cp_flag):
    cp_dir = '{:s}/checkpoints'.format(save_dir)
    if not os.path.exists(cp_dir):
        os.mkdir(cp_dir)
    filename = '{:s}/checkpoint2.pth.tar'.format(cp_dir)
    torch.save(state, filename)
    if cp_flag:
        shutil.copyfile(filename, '{:s}/checkpoint2_{:d}.pth.tar'.format(cp_dir, epoch+1))

def save_bestcheckpoint1(state, save_dir):
    cp_dir = '{:s}/checkpoints'.format(save_dir)
    torch.save(state, '{:s}/checkpoint1_0.pth.tar'.format(cp_dir))

def save_bestcheckpoint2(state, save_dir):
    cp_dir = '{:s}/checkpoints'.format(save_dir)
    torch.save(state, '{:s}/checkpoint2_0.pth.tar'.format(cp_dir))

def setup_logging(opt):
    mode = 'a' if opt.train['checkpoint'] else 'w'

    # create logger for training information
    logger = logging.getLogger('train_logger')
    logger.setLevel(logging.DEBUG)
    # create console handler and file handler
    console_handler = RichHandler(show_level=False, show_time=False, show_path=False)
    console_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler('{:s}/train_log.txt'.format(opt.train['save_dir']), mode=mode)
    file_handler.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(message)s')
    # add formatter to handlers
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    # add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # create logger for epoch results
    logger_results = logging.getLogger('results')
    logger_results.setLevel(logging.DEBUG)
    file_handler2 = logging.FileHandler('{:s}/epoch_results.txt'.format(opt.train['save_dir']), mode=mode)
    file_handler2.setFormatter(logging.Formatter('%(message)s'))
    logger_results.addHandler(file_handler2)

    logger.info('***** Training starts *****')
    logger.info('save directory: {:s}'.format(opt.train['save_dir']))
    if mode == 'w':
        logger_results.info('epoch\ttrain_loss\ttrain_loss_vor\ttrain_loss_cluster\ttrain_loss_ct\ttrain_loss_self\ttest_dice1\ttest_dice2')

    return logger, logger_results


if __name__ == '__main__':
    main()
