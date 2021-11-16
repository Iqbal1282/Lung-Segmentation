import torch
import torch.nn.functional as F
from tqdm import tqdm

from dice_score import multiclass_dice_coeff, dice_coeff


def evaluate(net, dataloader, device, transforms):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    dataloader.dataset.transforms = None 

    # iterate over the validation set
    for batch in dataloader:
        (image, mask_true) = batch
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.float32)
        #print(mask_true.shape)
        
        #mask_true = mask_true.permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)

            # convert to one-hot format
            mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
            # compute the Dice score
            #print(mask_pred.shape)
            dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)

    net.train()
    dataloader.dataset.transforms = transforms 

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches