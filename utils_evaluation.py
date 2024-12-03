import math
from tqdm import tqdm
from utils import calculate_iou
import torch


def logits2probs(logits):
    heads_prob = []
    if isinstance(logits, list):
        num_heads = len(logits)
        for head in range(num_heads):
            heads_prob.append(torch.sigmoid(logits[head]).squeeze())
    else:
        b_size, num_heads, H, W = logits.shape

        for head in range(num_heads):
            heads_prob.append(torch.sigmoid(logits[:,head]).squeeze())

    return heads_prob


def probs2segs(probs):
    num_heads = len(probs)
    heads_seg = []
    for head in range(num_heads):
        heads_seg.append((probs[head] >= 0.5).int())

    return heads_seg

# Function to handle predictions
def predict(model_id,logits):

    if model_id <10:  # vanilla U-Net
        probs = logits2probs(logits)
        final_pred = probs2segs(probs)[0]
    elif model_id == 10 or model_id == 11 or model_id == 12: #  BSNet
        probs = logits2probs(logits)
        final_pred = probs2segs(probs)[0]

    return final_pred


# the crop size (patch_size) is 256*256
# def extract_patches_testing(image, gt_mask, patch_size=256):
#     b, c, h, w = image.size()
#     patches = []
#     for image_id in range(b):
#         for i in range(0, h, patch_size):
#             for j in range(0, w, patch_size):
#                 img_patch = image[:, :, i:i+patch_size, j:j+patch_size]
#                 if img_patch.shape[2] == patch_size and img_patch.shape[3] == patch_size:
#                     gt_mask_patch = gt_mask[:, i:i + patch_size, j:j + patch_size]
#                     patches.append((img_patch,gt_mask_patch))
#
#     return patches
import torch.nn.functional as F

def extract_patches_testing(image, gt_mask, patch_size=256):
    """
    Extracts patches from the image and gt_mask, padding them to be divisible by patch_size.

    Args:
        image (torch.Tensor): The input image tensor of shape (b, c, h, w).
        gt_mask (torch.Tensor): The ground truth mask tensor of shape (b, h, w).
        patch_size (int): The size of each patch.

    Returns:
        patches (list): List of tuples containing image patches and corresponding gt_mask patches.
    """
    b, c, h, w = image.size()

    # Calculate padding to make image and gt_mask dimensions divisible by patch_size
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    padding = (0, pad_w, 0, pad_h)  # (left, right, top, bottom)

    # Apply padding
    padded_image = F.pad(image, padding, mode='constant', value=0)
    padded_mask = F.pad(gt_mask, padding, mode='constant', value=0)

    # Extract patches
    patches = [
        (padded_image[:, :, i:i+patch_size, j:j+patch_size],
         padded_mask[:, i:i+patch_size, j:j+patch_size])
        for i in range(0, h + pad_h, patch_size)
        for j in range(0, w + pad_w, patch_size)
    ]

    return patches

# I just removed bottom and right non-uniform batches 11 were removed!
# we are testing on the testing image (expect boundaries)
# TODO: it is just a placeholder
# def extract_patches_validation(image, gt_mask, patch_size=256):
#     b, c, h, w = image.size()
#     standard_patch_size_unit = 512# fix according to the protocol
#     num_patches_per_image = round(h/standard_patch_size_unit)**2
#     patches = []
#     for image_id in range(b):
#         for i in range(0, h, patch_size):
#             for j in range(0, w, patch_size):
#                 img_patch = image[:, :, i:i+patch_size, j:j+patch_size]
#                 if img_patch.shape[2] == patch_size and img_patch.shape[3] == patch_size:
#                     gt_mask_patch = gt_mask[:, i:i + patch_size, j:j + patch_size]
#                     patches.append((img_patch,gt_mask_patch))
#     # print(len(patches))
#     return patches

import torch



def extract_patches_validation(image: torch.Tensor, gt_mask: torch.Tensor, patch_size: int = 256) -> list:

    b, c, h, w = image.size()

    # Calculate necessary padding
    pad_h = (patch_size - (h % patch_size)) % patch_size
    pad_w = (patch_size - (w % patch_size)) % patch_size

    # Apply padding if necessary
    if pad_h > 0 or pad_w > 0:
        image = torch.nn.functional.pad(image, (0, pad_w, 0, pad_h), mode='constant', value=0)
        gt_mask = torch.nn.functional.pad(gt_mask, (0, pad_w, 0, pad_h), mode='constant', value=0)

    # Update dimensions after padding
    b, c, h, w = image.size()
    standard_patch_size_unit = 512  # fix according to the protocol
    # Calculate number of patches
    num_patches_per_image = round(h / standard_patch_size_unit) * round(w / standard_patch_size_unit)
    indices_per_image = {}
    # loop over the patch
    for image_id in range(b):
        available_indices = [(i, j) for i in range(0, h, patch_size) for j in range(0, w, patch_size)]
        random_indices = torch.randperm(len(available_indices))[:num_patches_per_image]
        indices_per_image[image_id] = []
        for idx in random_indices:
            i, j = available_indices[idx]
            indices_per_image[image_id].append((i, j))

    return indices_per_image,image,gt_mask, patch_size


from torch.cuda.amp import autocast

@torch.jit.script
def calculate_metrics(pred_mask_flat: torch.Tensor, gt_mask_flat: torch.Tensor):
    TP = (pred_mask_flat * gt_mask_flat).sum()
    FP = ((pred_mask_flat == 1) & (gt_mask_flat == 0)).sum()
    FN = ((pred_mask_flat == 0) & (gt_mask_flat == 1)).sum()
    TN = ((pred_mask_flat == 0) & (gt_mask_flat == 0)).sum()

    precision = TP / (TP + FP) if (TP + FP) > 0 else torch.tensor(0.0)
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else torch.tensor(0.0)
    recall = TP / (TP + FN) if (TP + FN) > 0 else torch.tensor(0.0)

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else torch.tensor(0.0)

    return precision, accuracy, f1, recall


def evaluate_model_testing(model, dataloader, args):
    model.eval()  # Set model to evaluation mode
    iou_list = []
    precision_list = []
    accuracy_list = []
    recall_list = []
    f1_list = []

    with torch.inference_mode():
        for images, labels in tqdm(dataloader):
            images = images.to(args.device)
            gt_mask = labels.to(args.device)

            # test on all the test image.
            data_pathches = extract_patches_testing(images,gt_mask)

            for image, gt_mask in data_pathches:
                # print("patch")
                with autocast():
                    logits = model(image.cuda().float())

                # the prediction is just 256x256 with two labels background (0) and building (1)
                pred_mask = predict(args.model_id,logits)

                # Compute IoU
                iou = calculate_iou(pred_mask, gt_mask)

                precision, accuracy, f1, recall = calculate_metrics(pred_mask.view(-1), gt_mask.reshape(-1))

                iou_list.append(iou)
                precision_list.append(precision)
                accuracy_list.append(accuracy)
                recall_list.append(recall)
                f1_list.append(f1)


    avg_iou = sum(iou_list) / len(iou_list) if len(iou_list) > 0 else 0
    avg_precision = sum(precision_list) / len(precision_list) if len(precision_list) > 0 else 0
    avg_accuracy = sum(accuracy_list) / len(accuracy_list) if len(accuracy_list) > 0 else 0
    avg_recall = sum(recall_list) / len(recall_list) if len(recall_list) > 0 else 0
    avg_f1 = sum(f1_list) / len(f1_list) if len(f1_list) > 0 else 0
    return avg_iou, avg_precision, avg_accuracy, avg_recall, avg_f1

# TODO
#model, valid_loader, args
def evaluate_model_validation(model, dataloader, args):
    model.eval()  # Set model to evaluation mode
    iou_list = []

    with torch.inference_mode():
        for images, labels in tqdm(dataloader):
            img_patches_batch = []
            gt_mask_patches_batch = []

            images = images.to(args.device)
            gt_mask = labels.to(args.device)
            # TODO [optimization] make faster
            data_patches_indices,images,gt_mask, patch_size = extract_patches_validation(images,gt_mask)

            for image_id in range(len(data_patches_indices)):
                crops_indices = data_patches_indices[image_id]
                for (i,j) in crops_indices:
                    img_patch = images[image_id, :, i:i + patch_size, j:j + patch_size].unsqueeze(0)  # Adding batch dimension
                    gt_mask_patch = gt_mask[image_id, i:i + patch_size, j:j + patch_size].unsqueeze(0)  # Adding batch dimension

                    # Accumulate patches
                    img_patches_batch.append(img_patch)
                    gt_mask_patches_batch.append(gt_mask_patch)

                    # Check if we have reached the desired batch size
                    if len(img_patches_batch) == args.batch_size:
                        # print("patch")
                        # with autocast():

                        logits = model(torch.cat(img_patches_batch, dim=0).cuda().float())

                        # the prediction is just 256x256 with two labels background (0) and building (1)
                        pred_mask = predict(args.model_id,logits)

                        # Compute IoU
                        iou = calculate_iou(pred_mask, torch.cat(gt_mask_patches_batch, dim=0))

                        iou_list.append(iou)
                        img_patches_batch = []
                        gt_mask_patches_batch = []



    avg_iou = sum(iou_list)/ len(iou_list) if len(iou_list) > 0 else 0
    return avg_iou

