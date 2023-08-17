import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import skimage.morphology as morph
import scipy.ndimage.morphology as ndi_morph
from skimage import measure
from scipy import misc
from model.modelW import ResWNet34
from model.model_WNet import WNet
import utils.utils as utils
from utils.accuracy import compute_metrics
import time
import imageio
from options import Options
from dataloaders.my_transforms import get_transforms
from tqdm import tqdm
from rich.table import Column, Table
from rich import print
import time


def main():
    opt = Options(isTrain=False)
    opt.parse()
    opt.save_options()

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in opt.test['gpus'])

    img_dir = opt.test['img_dir']
    imgh_dir = opt.test['imgh_dir']
    label_dir = opt.test['label_dir']
    save_dir = opt.test['save_dir']
    model_path1 = opt.test['model_path1']
    model_path2 = opt.test['model_path2']
    save_flag = opt.test['save_flag']

    # data transforms
    test_transform = get_transforms(opt.transform['test'])
    
    model1 = ResWNet34(seg_classes = 2, colour_classes = 3)
    model1 = torch.nn.DataParallel(model1)
    model1 = model1.cuda()
    model2 = WNet(n_channels = 1, seg_classes = 2, colour_classes = 3)
    # model2 = ResWNet34(seg_classes = 2, colour_classes = 3)
    model2 = torch.nn.DataParallel(model2)
    model2 = model2.cuda()
    cudnn.benchmark = True

    # ----- load trained model ----- #
    print("=> loading trained model")
    checkpoint1 = torch.load(model_path1)
    checkpoint2 = torch.load(model_path2)
    model1.load_state_dict(checkpoint1['state_dict'])
    model2.load_state_dict(checkpoint2['state_dict'])
    print("=> loaded model1 at epoch {}".format(checkpoint1['epoch']))
    print("=> loaded model2 at epoch {}".format(checkpoint2['epoch']))
    model1 = model1.module
    model2 = model2.module

    # switch to evaluate mode
    model1.eval()
    model2.eval()
    counter = 0
    print("=> Test begins:")

    img_names = os.listdir(img_dir)

    if save_flag:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        strs = img_dir.split('/')
        prob_maps_folder = '{:s}/{:s}_prob_maps'.format(save_dir, strs[-1])
        seg_folder = '{:s}/{:s}_segmentation'.format(save_dir, strs[-1])
        if not os.path.exists(prob_maps_folder):
            os.mkdir(prob_maps_folder)
        if not os.path.exists(seg_folder):
            os.mkdir(seg_folder)

    metric_names = ['acc', 'p_F1', 'dice', 'aji', 'dq', 'sq', 'pq']
    test_results = dict()
    all_result = utils.AverageMeter(len(metric_names))
    all_result1 = utils.AverageMeter(len(metric_names))
    all_result2 = utils.AverageMeter(len(metric_names))

    # calculte inference time
    time_list = []

    img_process = tqdm(img_names)
    for img_name in img_process:
        # load test image
        img_process.set_description('=> Processing image {:s}'.format(img_name))
        img_path = '{:s}/{:s}'.format(img_dir, img_name)
        imgh_path = '{:s}/{:s}'.format(imgh_dir, img_name)
        img = Image.open(img_path)
        imgh = Image.open(imgh_path)
        import ipdb; ipdb.set_trace()
        ori_h = img.size[1]
        ori_w = img.size[0]
        name = os.path.splitext(img_name)[0]
        label_path = '{:s}/{:s}_label.png'.format(label_dir, name)
        gt = imageio.imread(label_path)
     
        input = test_transform((img,))[0].unsqueeze(0)
        inputh = test_transform((imgh,))[0].unsqueeze(0)

        img_process.set_description('Computing output probability maps...')
        begin = time.time()
        prob_maps, prob_maps1, prob_maps2 = get_probmaps(input, inputh, model1, model2, opt)
        end = time.time()
        time_list.append(end - begin)
        pred = np.argmax(prob_maps, axis=0)  # prediction

        pred_labeled = measure.label(pred)
        pred_labeled = morph.remove_small_objects(pred_labeled, opt.post['min_area'])
        pred_labeled = ndi_morph.binary_fill_holes(pred_labeled > 0)
        pred_labeled = measure.label(pred_labeled)

        img_process.set_description('Computing metrics...')
        metrics = compute_metrics(pred_labeled, gt, metric_names)
        print(metrics)

        test_results[name] = [metrics[m] for m in metric_names]
        all_result.update([metrics[m] for m in metric_names])
        
        pred1 = np.argmax(prob_maps1, axis=0)  # prediction                         
        pred_labeled1 = measure.label(pred1)
        pred_labeled1 = morph.remove_small_objects(pred_labeled1, opt.post['min_area'])
        pred_labeled1 = ndi_morph.binary_fill_holes(pred_labeled1 > 0)
        pred_labeled1 = measure.label(pred_labeled1)
        metrics = compute_metrics(pred_labeled1, gt, metric_names)
        all_result1.update([metrics[m] for m in metric_names])

        pred = np.argmax(prob_maps2, axis=0)  # prediction                         
        pred_labeled2 = measure.label(pred)
        pred_labeled2 = morph.remove_small_objects(pred_labeled2, opt.post['min_area'])
        pred_labeled2 = ndi_morph.binary_fill_holes(pred_labeled2 > 0)
        pred_labeled2 = measure.label(pred_labeled2)
        metrics = compute_metrics(pred_labeled2, gt, metric_names)
        all_result2.update([metrics[m] for m in metric_names])

        # save image
        if save_flag:
            img_process.set_description('Saving image results...')
            imageio.imsave('{:s}/{:s}_pred.png'.format(prob_maps_folder, name), pred.astype(np.uint8) * 255)
            imageio.imsave('{:s}/{:s}_prob.png'.format(prob_maps_folder, name), (prob_maps[1] * 255).astype(np.uint8))
            final_pred = Image.fromarray(pred_labeled.astype(np.uint16))
            final_pred.save('{:s}/{:s}_seg.tiff'.format(seg_folder, name))

            # save colored objects
            pred_colored_instance = np.zeros((ori_h, ori_w, 3))
            for k in range(1, pred_labeled.max() + 1):
                pred_colored_instance[pred_labeled == k, :] = np.array(utils.get_random_color())
            filename = '{:s}/{:s}_seg_colored.png'.format(seg_folder, name)
            imageio.imsave(filename, (pred_colored_instance * 255).astype(np.uint8))

        counter += 1

    print('=> Processed all {:d} images'.format(counter))
    print('=> Average time per image: {:.4f} s'.format(np.mean(time_list)))
    print('=> Std time per image: {:.4f} s'.format(np.std(time_list)))

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Model", style="dim", width=12)
    table.add_column("Acc")
    table.add_column("F1")
    table.add_column("Dice")
    table.add_column("AJI")
    table.add_column("DQ")
    table.add_column("SQ")
    table.add_column("PQ")
    a, a1, a2 = all_result.avg, all_result1.avg, all_result2.avg
    table.add_row('ens', f'{a[0]: .4f}', f'{a[1]: .4f}', f'{a[2]: .4f}', f'{a[3]: .4f}', f'{a[4]: .4f}', f'{a[5]: .4f}', f'{a[6]: .4f}'.format(a=all_result.avg))
    table.add_row('m1', f'{a1[0]: .4f}', f'{a1[1]: .4f}', f'{a1[2]: .4f}', f'{a1[3]: .4f}', f'{a1[4]: .4f}', f'{a1[5]: .4f}', f'{a1[6]: .4f}'.format(a1=all_result1.avg))
    table.add_row('m2', f'{a2[0]: .4f}', f'{a2[1]: .4f}', f'{a2[2]: .4f}', f'{a2[3]: .4f}', f'{a2[4]: .4f}', f'{a2[5]: .4f}', f'{a2[6]: .4f}'.format(a2=all_result2.avg))
    print(table)

    header = metric_names
    utils.save_results(header, all_result.avg, test_results, f'{save_dir}/test_results_{checkpoint1["epoch"]}_{checkpoint2["epoch"]}_{all_result.avg[5]:.4f}_{all_result1.avg[5]:.4f}_{all_result2.avg[5]:.4f}.txt')


def get_probmaps(input, inputh, model1, model2, opt):
    size = opt.test['patch_size']
    overlap = opt.test['overlap']
    output1 = split_forward1(model1, inputh, size, overlap)
    output1 = output1.squeeze(0)
    prob_maps1 = F.softmax(output1, dim=0).cpu().numpy()

    output2 = split_forward1(model2, inputh, size, overlap)
    output2 = output2.squeeze(0)
    prob_maps2 = F.softmax(output2, dim=0).cpu().numpy()

    prob_maps = (prob_maps1 + prob_maps2) / 2.0

    return prob_maps, prob_maps1, prob_maps2


def split_forward1(model, input, size, overlap, outchannel = 2):
    '''
    split the input image for forward passes
    motification: if single image, split it into patches and concat, forward once.
    '''

    b, c, h0, w0 = input.size()

    # zero pad for border patches
    pad_h = 0
    if h0 - size > 0 and (h0 - size) % (size - overlap) > 0:
        pad_h = (size - overlap) - (h0 - size) % (size - overlap)
        tmp = torch.zeros((b, c, pad_h, w0))
        input = torch.cat((input, tmp), dim=2)

    if w0 - size > 0 and (w0 - size) % (size - overlap) > 0:
        pad_w = (size - overlap) - (w0 - size) % (size - overlap)
        tmp = torch.zeros((b, c, h0 + pad_h, pad_w))
        input = torch.cat((input, tmp), dim=3)

    _, c, h, w = input.size()

    output = torch.zeros((input.size(0), outchannel, h, w))
    input_vars = []
    for i in range(0, h-overlap, size-overlap):
        r_end = i + size if i + size < h else h
        ind1_s = i + overlap // 2 if i > 0 else 0
        ind1_e = i + size - overlap // 2 if i + size < h else h
        for j in range(0, w-overlap, size-overlap):
            c_end = j+size if j+size < w else w

            input_patch = input[:,:,i:r_end,j:c_end]
            # input_var = input_patch.cuda()
            input_var = input_patch.numpy()
            input_vars.append(input_var)
    input_vars = torch.as_tensor(input_vars)
    input_vars = input_vars.squeeze(1).cuda()
    with torch.no_grad():
        output_patches, _ = model(input_vars)
    idx = 0
    for i in range(0, h-overlap, size-overlap):
        r_end = i + size if i + size < h else h
        ind1_s = i + overlap // 2 if i > 0 else 0
        ind1_e = i + size - overlap // 2 if i + size < h else h
        for j in range(0, w-overlap, size-overlap):
            c_end = j+size if j+size < w else w
            output_patch = output_patches[idx]
            idx += 1
            output_patch = output_patch.unsqueeze(0)
            ind2_s = j+overlap//2 if j>0 else 0
            ind2_e = j+size-overlap//2 if j+size<w else w
            output[:,:,ind1_s:ind1_e, ind2_s:ind2_e] = output_patch[:,:,ind1_s-i:ind1_e-i, ind2_s-j:ind2_e-j]
    output = output[:,:,:h0,:w0].cuda()

    return output


if __name__ == '__main__':
    main()
