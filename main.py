# v.4
import sys
import torch
import argparse
from modules import BSNet
from utils_test import  run_test
from utils_train import run_train
from datasets import select_dataset
from utils_UI import print_error_message
import segmentation_models_pytorch as smp
from utils import set_seed, logs, get_loss_metrics, get_path_batch_size, check_system_validity


run_modes_names = ["train", "test", "continue training"]
ds_names = ["Massa", "WHU", "Inria", "GBSS"]
ds_preprocessing = ["random cropping", "selected cropping"]
models_names = ["FPN", "Unet", "UNet++", "DeepLabV3+", "PSPNet", "UANet", "6 NO", "7 NO", "8 NO", "9 NO", "BSNet"]


# Initialize the argument parser
parser = argparse.ArgumentParser(description="Set training/testing parameters from the terminal")

# Add arguments
parser.add_argument('--run_mode', type=int, choices=[0, 1, 2], help="0 for training, 1 for testing, "
                                                                       "2 for continue training")
parser.add_argument('--epochs', type=int, default=200, help="Number of training epochs (default: 200)")
parser.add_argument('--ds_id', type=int, choices=[0, 1, 2, 3], default=0, help="Dataset ID")
parser.add_argument('--model_id', type=int, default=0, help="Model ID: Use 10 for our model: BSNet")
parser.add_argument('--train_ds_proc_id', type=int, choices=[0, 1], default=1, help="Training data processing ID")
parser.add_argument('--batch_size', type=int, default=-1, help="Batch size")
parser.add_argument('--itr_num', type=int, default=0, help="The iteration index")
parser.add_argument('--patience', type=int, default=20, help="patience num of epochs to wait")
parser.add_argument('--num_workers', type=int, default=16, help="Number of workers for data loading")
parser.add_argument('--MAX_NUM_CROPS', type=int, default=50, help="Maximum number of crops")
parser.add_argument('--min_pixel_count', type=int, default=256*256*0.0, help="Minimum pixel count")
parser.add_argument('--max_pixel_count', type=int, default=256*256*1.0, help="Maximum pixel count")
parser.add_argument('--islog2neptune', type=bool, default=False, help="Flag to log results to neptune")
parser.add_argument('--visualize_image', type=bool, default=False, help="Flag to visualize the full image")
parser.add_argument('--visualize_crops', type=bool, default=False, help="Flag to visualize the cropped images")
parser.add_argument('--visualize_segmentation_pred', type=bool, default=False, help="Flag to visualize the segmentation predictions")
parser.add_argument('--continue_train_model_path', type=str, default="No check point", help="Path to the model checkpoint for continuing training")
parser.add_argument('--folder_name', type=str, default="check_points/bs_lines", help="{folder_name}/bst_mdl_{model_id}_{ds_id}_{train_ds_proc_id}.pth")


# Check if arguments were provided in the command line
if len(sys.argv) > 1:
    # Parse arguments from the command line
    args = parser.parse_args()
else: # No command-line arguments provided, set default values manually
    class Args:
        def __init__(self):
            self.run_mode = 0 # 0 train; 1 test, 2 continue training
            self.epochs = 200
            self.ds_id = 0  # 0->Massa 1->WHU, 2->Inria, 3->GBSS
            self.model_id = 10  # 0=>"FPN", 1=>"Unet", 2=>"UNet++", 3=>"DeepLabV3+", 4=>"PSPNet",  5=>"UANet", "6 NO", "7 NO", "8 NO", "9 NO", 10=>"BSNet"
            self.train_ds_proc_id = 1  # 0: random cropping, 1: selected cropping
            self.batch_size = 4
            self.itr_num = 0
            self.patience = 200  # Number of epochs to wait before stopping if no improvement
            self.num_workers = 8
            self.MAX_NUM_CROPS = 50
            self.min_pixel_count = 256*256*0.0
            self.max_pixel_count = 256*256*1.0
            self.visualize_image = False
            self.visualize_crops = False
            self.visualize_segmentation_pred = False
            self.islog2neptune = False
            self.continue_train_model_path ="None" #"check_points/ours/bst_mdl_0_4_1.pth"
            self.folder_name = "check_points/bs_lines"#"check_points/bs_lines"

    args = Args()

set_seed(42*args.itr_num)

#Specifically for UANet, we call their train/test modules
if  args.model_id == 5:
    print_error_message("please run Uncertainty-aware-Network Code!", 1)

# Conditional check to ensure --folder_name is required only when --run_mode is 1
if args.run_mode == 1 and not args.folder_name:
    parser.error('--folder_name is required when --run_mode is 1 for testing.')

args = get_path_batch_size(args)
args.run_modes_names = run_modes_names
args.ds_preprocessing = ds_preprocessing
args.ds_names = ds_names
args.models_names = models_names
args.folder_root = ""


#select dataset
train_dataset, valid_dataset, tst_dataset, args, train_loader, valid_loader, tst_loader = select_dataset(args)

loss, metrics = get_loss_metrics(smp)

args = check_system_validity(args)

if args.model_id < 10:
    if args.model_id == 0:
        model = smp.FPN('resnet50', in_channels=3)
    elif args.model_id == 1:
        model = smp.Unet('resnet50', in_channels=3)
    elif args.model_id == 2:
        model = smp.UnetPlusPlus('resnet50', in_channels=3)
    elif args.model_id == 3:
        model = smp.DeepLabV3Plus('resnet50', in_channels=3)
    elif args.model_id == 4:
        model = smp.PSPNet('resnet50', in_channels=3)
    else:
        print_error_message("Issue in the model id! Check baseline id", True)

    model.cuda()
    # define optimizer
    args.lr = 0.0005
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr)

    lr_scheduler = None


elif args.model_id == 10:
    # Get our model
    model = BSNet()

    # Define parameter groups with different learning rates
    args.lr = 0.0005
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr)
    lr_scheduler = None
else:
    print_error_message("Issue in the model id!", True)

run = logs(args)

if args.run_mode == 0: # training
    run_train(model, train_loader, optimizer, loss, valid_loader, train_dataset, run, lr_scheduler, metrics, args)
elif args.run_mode == 1: # testing
   run_test(model,tst_loader, args)
elif args.run_mode == 2: # continue training
    checkpoint = torch.load(args.continue_train_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    run_train(model, train_loader, optimizer, loss, valid_loader, train_dataset, run, lr_scheduler, metrics, args)
