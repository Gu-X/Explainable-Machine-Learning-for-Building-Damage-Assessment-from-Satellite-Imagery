import os
import random
import numpy as np
import torch
import neptune

from utils_UI import print_info_message, print_error_message



def set_seed(seed):
    # Set the seed for random number generation in Python
    random.seed(seed)

    # Set the seed for random number generation in NumPy
    np.random.seed(seed)

    # Set the seed for random number generation in PyTorch
    torch.manual_seed(seed)

    # If using CUDA, set the seed for all GPUs
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Ensure that all operations are deterministic on GPU (if needed)
    torch.backends.cudnn.deterministic = True

    # Optionally disable CUDA benchmark mode for reproducibility
    torch.backends.cudnn.benchmark = False





# Perform one hot encoding on label
def one_hot_encode_xxxx(label, label_values):
    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes
    # Arguments
        label: The 2D array segmentation image label
        label_values

    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of num_classes
    """

    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)
    return semantic_map.astype('float')


# Perform colour coding on the reverse-one-hot outputs
def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.
    # Arguments
        image: single channel array where each value represents the class key.
        label_values

    # Returns
        Colour coded image for segmentation visualization
    """
    colour_codes = np.array([[0,0,0],[0, 255,0],[255,200,50],[255,0,0],[0,0,255],[255,255,255]])
    image[image==255] = 5
    # colour_codes = np.array(label_values)

    x = colour_codes[image]

    return x


@torch.jit.script
def calculate_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
    """
    Calculate the Intersection over Union (IoU) between two binary images.

    Args:
        pred_mask (torch.Tensor): First binary image.
        gt_mask (torch.Tensor): Second binary image.

    Returns:
        torch.Tensor: IoU score between the two binary images.
    """
    # Ensure both masks are treated as boolean
    pred_mask = pred_mask > 0  # Treat any non-zero as True
    gt_mask = gt_mask > 0      # Treat any non-zero as True

    # Check if dimensions match
    if pred_mask.size() != gt_mask.size():
        print(pred_mask.size())
        print(gt_mask.size())
        raise ValueError("Prediction and ground truth masks must have the same dimensions.")

    intersection = torch.logical_and(pred_mask, gt_mask).sum()
    union = torch.logical_or(pred_mask, gt_mask).sum()

    # Calculate IoU
    iou = intersection / union if union > 0 else torch.tensor(0.0)

    return iou


def logs(args):

    # print_info_message(
    #     f"Info: ds_name: {args.ds_names[args.ds_id]}, version: {args.ds_preprocessing[args.train_ds_proc_id]}, model:{args.models_names[args.model_id]},"
    #     f" MAX_NUM_CROPS: {args.MAX_NUM_CROPS} ")

    # Print the information
    print_info_message(
        f"Info: run_mode: {args.run_modes_names[args.run_mode]}, epochs: {args.epochs}, ds_name: {args.ds_names[args.ds_id]}, "
        f"model: {args.models_names[args.model_id]}, train_ds_processing: {args.ds_preprocessing[args.train_ds_proc_id]}, "
        f"batch_size: {args.batch_size}, num_workers: {args.num_workers}, \nMAX_NUM_CROPS: {args.MAX_NUM_CROPS}, "
        f"min_pixel_count: {args.min_pixel_count}, max_pixel_count: {args.max_pixel_count}, "
        f"visualize_image: {args.visualize_image}, visualize_crops: {args.visualize_crops}, "
        f"visualize_segmentation_pred: {args.visualize_segmentation_pred}, \nislog2neptune: {args.islog2neptune}, "
        f"continue_train_model_path: {args.continue_train_model_path}"
    )

    if not args.islog2neptune:
        return None
    params = {
        "batch_size": args.batch_size,
        "run_mode": args.run_modes_names[args.run_mode],
        "learning_rate": args.lr,
        "architecture": args.models_names[args.model_id],
        "dataset": args.ds_names[args.ds_id],
        "epochs": args.epochs,
        "preprocessing": args.ds_preprocessing[args.train_ds_proc_id] + "90% Selective and 10% Rand"
    }

    test_or_train = "---train" if args.run_mode == 0 else "---continue train" if args.run_mode == 2 else "---test"


    run = neptune.init_run(
        name= args.models_names[args.model_id] + "-" + args.ds_names[args.ds_id] + "-" + args.ds_preprocessing[args.train_ds_proc_id] \
             + test_or_train,
        project="XML-Alpha/XML-AISurrey",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmZDIzYmQxOS01YWI2LTQwZGQtODBkOS0yOTAxNDEwNzUwZjgifQ==",
    )  # your credentials

    run["sys/tags"].add(test_or_train[3:])
    run["parameters"] = params


    return run

def get_loss_metrics(smp):
    # define loss function
    loss_BCE = smp.utils.losses.BCELoss()
    loss_Dice = smp.utils.losses.DiceLoss()

    loss = loss_Dice + loss_BCE

    # define metrics
    metrics = [smp.utils.metrics.IoU(threshold=0.5, activation=None)]
    return loss, metrics


def get_path_batch_size(args):
    args.path2save_model = f'{args.folder_name}/bst_mdl_{args.model_id}_{args.ds_id}_{args.train_ds_proc_id}_{args.itr_num}.pth'
    args.test_model_path = f'{args.folder_name}/bst_mdl_{args.model_id}_{args.ds_id}_{args.train_ds_proc_id}_{args.itr_num}.pth'
    args.path2save_test_results = f"paper_test_results_{args.itr_num}.txt"
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if "mnt/fast/nobackup" in dir_path:
        args.isAISurrey = True
    else:
        args.isAISurrey = False

    # system_is = "xML_Alpha"
    args.tr_batch_size = args.vl_batch_size = args.tst_batch_size = args.batch_size
    if args.isAISurrey:
        args.path2save_test_results = f'/mnt/fast/nobackup/users/ak0084/Code/xML_Alpha/paper_test_results_{args.itr_num}.txt'
        args.path2save_model = f'/mnt/fast/nobackup/users/ak0084/Code/xML_Alpha/{args.folder_name}/' \
                          f'bst_mdl_{args.model_id}_{args.ds_id}_{args.train_ds_proc_id}_{args.itr_num}.pth'
        args.test_model_path = f'/mnt/fast/nobackup/users/ak0084/Code/xML_Alpha/{args.folder_name}/' \
                          f'bst_mdl_{args.model_id}_{args.ds_id}_{args.train_ds_proc_id}_{args.itr_num}.pth'
        args.tr_batch_size = args.vl_batch_size = args.tst_batch_size = args.batch_size  # was 8 for all other exprs change for xbd ours to make 64, inria 32
    return args

def check_system_validity(args):
    # Set device: `cuda` or `cpu`
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.device.type != "cuda":
        print_error_message("Device is NOT cuda", True)

    return args