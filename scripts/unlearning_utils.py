import torch
import torch.nn.functional as F
from torch import nn


import torch
import torch.nn.functional as F
from torch import nn

class SSIMLoss(nn.Module):
    def __init__(self, window_size=3, size_average=False, size_sum=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.size_sum = size_sum

    def gaussian_window(self, channel, window_size, sigma=1.5):
        _1D_window = torch.exp(-torch.tensor([(x - window_size//2)**2/float(2*sigma**2) for x in range(window_size)]))
        _1D_window /= _1D_window.sum()
        _2D_window = _1D_window.unsqueeze(1).mm(_1D_window.unsqueeze(0))
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def ssim(self, img1, img2, window, window_size, channel, size_average=False, size_sum=True):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average and (size_sum != True):
            return ssim_map.mean()
        elif size_sum == True:
            return ssim_map.sum()
        else:
            return ssim_map.mean([1, 2, 3])

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        window = self.gaussian_window(channel, self.window_size).to(img1.device)
        return 1 - self.ssim(img1, img2, window, self.window_size, channel, self.size_average)


def count_changed_pixels(img1, img2, threshold=0.1):
    """
    Count the number of pixels that are different between two images.

    Parameters:
    img1 (torch.Tensor): The first image tensor.
    img2 (torch.Tensor): The second image tensor.
    threshold (float): The threshold to consider a pixel as changed.

    Returns:
    int: The number of changed pixels.
    """

    # Calculate the absolute difference between the two images
    diff = torch.abs(img1 - img2)

    # Apply the threshold
    if threshold > 0:
        changed_pixels = torch.sum(diff > threshold)
    else:
        changed_pixels = torch.sum(diff != 0)

    return changed_pixels.item()
    

def config_args(args):
    if args.ablation_mode == None:
        if args.specific_obj == 48:
            args.iteration=1
            args.scale=10000
            args.damp=0

        if args.specific_obj == 20:
            args.iteration=1
            args.scale=1000
            args.damp=0

        if args.specific_obj == 3:

            args.iteration=1
            args.scale=1000
            args.damp=0

            # args.iteration=1
            # args.scale=1000
            # args.damp=0
    
    if 'decoder' in args.unlearn_module:
        args.scale=2000



