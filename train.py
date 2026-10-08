import os
import torch
from data import train_dataloader
from utils import Adder, Timer, check_lr
from torch.utils.tensorboard import SummaryWriter
from valid import _valid
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models

import math

from warmup_scheduler import GradualWarmupScheduler

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        # torchvision 新版用 weights=...，旧版用 pretrained=True
        try:
            vgg_pretrained_features = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        except Exception:
            vgg_pretrained_features = models.vgg19(pretrained=True).features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()

        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]


class ContrastLoss(nn.Module):
    """
    a: anchor, p: positive, n: negative
    默认用 d_ap / (d_an + eps)
    """
    def __init__(self, device, ablation=False):
        super(ContrastLoss, self).__init__()
        self.vgg = Vgg19(requires_grad=False).to(device).eval()
        self.l1 = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.ab = ablation

        # ImageNet mean/std for VGG
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        self.register_buffer("vgg_mean", mean)
        self.register_buffer("vgg_std", std)

    def _prep(self, x):
        # x: (B,C,H,W), assume in [0,1] or [0,255] normalized already by your pipeline
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] != 3:
            # 强制取前3通道
            x = x[:, :3, :, :]

        # 保险：把范围 clamp 到 [0,1]
        x = x.clamp(0.0, 1.0)
        x = (x - self.vgg_mean) / self.vgg_std
        return x

    def forward(self, a, p, n):
        a = self._prep(a)
        p = self._prep(p)
        n = self._prep(n)

        a_vgg, p_vgg, n_vgg = self.vgg(a), self.vgg(p), self.vgg(n)

        loss = 0.0
        for i in range(len(a_vgg)):
            d_ap = self.l1(a_vgg[i], p_vgg[i].detach())
            if not self.ab:
                d_an = self.l1(a_vgg[i], n_vgg[i].detach())
                #contrastive = d_ap / (d_an + 1e-7)
                contrastive = torch.relu(d_ap - d_an + 0.5)
            else:
                contrastive = d_ap
            loss = loss + self.weights[i] * contrastive
        return loss

def _train(model, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.L1Loss()
    contrast_criterion = ContrastLoss(device=device, ablation=getattr(args, "contrast_ablation", False))
    lambda_contrast = args.lambda_contrast


    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8)
    dataloader = train_dataloader(args.data_dir, args.batch_size, args.num_worker, args.data)
    max_iter = len(dataloader)
    warmup_epochs = 3
    eta_min = 1e-6

    def set_epoch_lr(optimizer, epoch_idx):
        if epoch_idx <= warmup_epochs:
            # 3 epoch warm-up
            lr = args.learning_rate * epoch_idx / warmup_epochs
        else:
            # cosine annealing
            t = epoch_idx - warmup_epochs
            T_max = args.num_epoch - warmup_epochs

            lr = eta_min + (args.learning_rate - eta_min) * \
                (1.0 + math.cos(math.pi * t / T_max)) / 2.0

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return lr
    epoch = 1

    if args.resume:
        state = torch.load(args.resume, map_location=device)

        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])

        saved_epoch = state['epoch']
        epoch = saved_epoch + 1

        print('Resume from %d, continue at %d'
            % (saved_epoch, epoch))

    writer = SummaryWriter()
    epoch_pixel_adder = Adder()
    epoch_fft_adder = Adder()
    iter_pixel_adder = Adder()
    iter_fft_adder = Adder()
    epoch_contrast_adder = Adder()
    iter_contrast_adder = Adder()
    epoch_timer = Timer('m')
    iter_timer = Timer('m')
    best_psnr=-1

    for epoch_idx in range(epoch, args.num_epoch + 1):

        current_lr = set_epoch_lr(optimizer, epoch_idx)

        print("Epoch %d start, LR = %.10f" %
            (epoch_idx, current_lr))

        epoch_timer.tic()
        iter_timer.tic()
        for iter_idx, batch_data in enumerate(dataloader):

            input_img, label_img = batch_data
            input_img = input_img.to(device)
            label_img = label_img.to(device)

            optimizer.zero_grad()
            pred_img = model(input_img)
            label_img2 = F.interpolate(label_img, scale_factor=0.5, mode='bilinear')
            label_img4 = F.interpolate(label_img, scale_factor=0.25, mode='bilinear')
            l1 = criterion(pred_img[0], label_img4)
            l2 = criterion(pred_img[1], label_img2)
            l3 = criterion(pred_img[2], label_img)
            loss_content = l1+l2+l3

            label_fft1 = torch.fft.fft2(label_img4, dim=(-2,-1))
            label_fft1 = torch.stack((label_fft1.real, label_fft1.imag), -1)

            pred_fft1 = torch.fft.fft2(pred_img[0], dim=(-2,-1))
            pred_fft1 = torch.stack((pred_fft1.real, pred_fft1.imag), -1)

            label_fft2 = torch.fft.fft2(label_img2, dim=(-2,-1))
            label_fft2 = torch.stack((label_fft2.real, label_fft2.imag), -1)

            pred_fft2 = torch.fft.fft2(pred_img[1], dim=(-2,-1))
            pred_fft2 = torch.stack((pred_fft2.real, pred_fft2.imag), -1)

            label_fft3 = torch.fft.fft2(label_img, dim=(-2,-1))
            label_fft3 = torch.stack((label_fft3.real, label_fft3.imag), -1)

            pred_fft3 = torch.fft.fft2(pred_img[2], dim=(-2,-1))
            pred_fft3 = torch.stack((pred_fft3.real, pred_fft3.imag), -1)

            f1 = criterion(pred_fft1, label_fft1)
            f2 = criterion(pred_fft2, label_fft2)
            f3 = criterion(pred_fft3, label_fft3)
            loss_fft = f1+f2+f3

            # --- contrastive loss ---
            # anchor: pred (full resolution), positive: gt, negative: shuffled gt in batch
            #a = pred_img[2]
            #p = label_img
            #n = torch.roll(label_img, shifts=1, dims=0)  # batch 内错位当负样本

            #loss_contrast = contrast_criterion(a, p, n)
            # --- contrastive loss ---
            a = pred_img[2]
            p = label_img

            if label_img.size(0) > 1:
                n = torch.roll(label_img, shifts=1, dims=0)
                loss_contrast = contrast_criterion(a, p, n)
            else:
                # batch_size=1 时 roll 后 n==p，会退化；直接跳过
                loss_contrast = torch.zeros((), device=device)

            #loss = loss_content + 0.1 * loss_fft
            loss = loss_content + args.fft_weight * loss_fft + lambda_contrast * loss_contrast
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.001)
            #torch.nn.utils.clip_grad_norm_(model.parameters(), getattr(args, "grad_clip", 0.1))
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            iter_pixel_adder(loss_content.item())
            iter_fft_adder(loss_fft.item())
            iter_contrast_adder(loss_contrast.item())

            epoch_pixel_adder(loss_content.item())
            epoch_fft_adder(loss_fft.item())
            epoch_contrast_adder(loss_contrast.item())

            if (iter_idx + 1) % args.print_freq == 0:
                print("Time: %7.4f Epoch: %03d Iter: %4d/%4d LR: %.10f Loss content: %7.4f Loss fft: %7.4f Loss contrast: %7.4f" % (
                    iter_timer.toc(), epoch_idx, iter_idx + 1, max_iter, optimizer.param_groups[0]['lr'], iter_pixel_adder.average(),
                    iter_fft_adder.average(), iter_contrast_adder.average()))
                writer.add_scalar('Contrast Loss', loss_contrast.item(), iter_idx + (epoch_idx-1)*max_iter)
                writer.add_scalar('Pixel Loss', iter_pixel_adder.average(), iter_idx + (epoch_idx-1)* max_iter)
                writer.add_scalar('FFT Loss', iter_fft_adder.average(), iter_idx + (epoch_idx - 1) * max_iter)
                
                iter_timer.tic()
                iter_pixel_adder.reset()
                iter_fft_adder.reset()
                iter_contrast_adder.reset()
        overwrite_name = os.path.join(args.model_save_dir, 'model.pkl')
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch_idx
        }, overwrite_name)

        if epoch_idx % args.save_freq == 0:
            save_name = os.path.join(args.model_save_dir, 'model_%d.pkl' % epoch_idx)
            torch.save({'model': model.state_dict()}, save_name)
        print("EPOCH: %02d\nElapsed time: %4.2f Epoch Pixel Loss: %7.4f Epoch FFT Loss: %7.4f Epoch Contrast Loss: %7.4f" % (
            epoch_idx, epoch_timer.toc(), epoch_pixel_adder.average(), epoch_fft_adder.average(), epoch_contrast_adder.average()))
        epoch_fft_adder.reset()
        epoch_pixel_adder.reset()
        epoch_contrast_adder.reset()
        #scheduler.step()
        if epoch_idx % args.valid_freq == 0:
            val = _valid(model, args, epoch_idx)
            print('%03d epoch \n Average PSNR %.2f dB' % (epoch_idx, val))
            writer.add_scalar('PSNR', val, epoch_idx)
            if val >= best_psnr:
                torch.save({'model': model.state_dict()}, os.path.join(args.model_save_dir, 'Best.pkl'))
    save_name = os.path.join(args.model_save_dir, 'Final.pkl')
    torch.save({'model': model.state_dict()}, save_name)