import torch
import numpy as np
import torch.nn.functional as F
from scipy.ndimage import sobel
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from utils_visualize import visualize
from utils import colour_code_segmentation
from segmentation_models_pytorch import utils as smp_utils
from utils_UI import print_info_message, print_error_message
from utils_evaluation import logits2probs, probs2segs, predict, evaluate_model_validation


def compute_edge_mask(gt_mask):
    # Apply Sobel filter to get edge map
    edges = sobel(gt_mask.cpu().numpy().astype(float))
    edges = (edges > 0).astype(float)
    return torch.tensor(edges, dtype=torch.float32).to(gt_mask.device)

def edge_aware_loss(pred, gt):
    edge_mask = compute_edge_mask(gt)
    edge_loss = F.binary_cross_entropy(pred, gt, weight=edge_mask)
    return edge_loss

def total_variation_loss(x):
    diff_i = x[:, :, 1:, :] - x[:, :, :-1, :]
    diff_j = x[:, :, :, 1:] - x[:, :, :, :-1]
    return torch.sum(torch.abs(diff_i)) + torch.sum(torch.abs(diff_j))



def create_custom_cmap():
    """
    Creates a custom colormap for visualizing binary predictions.
    Dark blue represents values close to 0, dark red represents values close to 1.
    """
    # colors = [(0, 0, 1, alpha) for alpha in np.linspace(0, 0.5, 128)] + [(1, 0, 0, 1 - alpha) for alpha in np.linspace(0.5, 1, 128)]
    colors = [
                 (0, 0, alpha, 1) for alpha in np.linspace(0, 1, 128)
             ] + [
                 (1, alpha, 0, 1) for alpha in np.linspace(0, 1, 128)
             ]
    cmap = mcolors.LinearSegmentedColormap.from_list('binary_cmap', colors, N=256)
    return cmap





def find_intersection(C0, C1, weight):
    result = torch.full(C0.shape, 0, dtype=torch.float).cuda()
    result[(C0 > 0) & (C1 > 0)] = weight
    result[(C0 < 0) & (C1 < 0)] = -1 * weight

    return result


def visualize_predictions(predictions, ground_truth, figure_title):
    """
    Visualizes binary mask predictions alongside the ground truth.

    :param predictions: List of prediction arrays, each being a 2D numpy array with values between 0 and 1.
    :param ground_truth: 2D numpy array of ground truth with values 0 or 1.
    """

    num_heads = len(predictions)
    fig, axes = plt.subplots(1, num_heads + 1, figsize=(15, 5))
    fig.suptitle(figure_title)  # Add title to the entire figure
    # Create the custom colormap
    cmap = create_custom_cmap()

    heads_names = ["lw_lvl_hd", "md_lvl_seg_stg1_hd", "md_lvl_seg_stg2_hd", "hgh_lvl_hd", "Fused Prediction"]
    # Display predictions
    for i in range(num_heads):
        ax = axes[i]
        ax.imshow(predictions[i][0].detach().cpu(), cmap=cmap, vmin=0, vmax=1)
        ax.set_title(f'{heads_names[i]}')
        # plt.colorbar(im, ax=ax)

    # Display ground truth
    ax = axes[num_heads]
    im = ax.imshow(ground_truth[0], cmap=cmap, vmin=0, vmax=1)
    ax.set_title('Ground Truth')

    # plt.colorbar(im, ax=ax)

    # Adjust layout to avoid the rightmost image getting smaller
    fig.subplots_adjust(right=0.85)

    # Add a single color bar for the entire figure, located outside the grid
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # Adjust the position of the color bar
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Probability')

    plt.show()


# this should be called only for model id =1 (train or valid)
def calculate_loss_BSNet(running_mode, model_id, loss, fuzed_seg_and_logits, grnd_truth, loss_weights, figure_title=None):
    fuzed_logits = fuzed_seg_and_logits[0]


    fuzed_prob = logits2probs(fuzed_logits)

    # Calculate individual losses
    loss_fused_seg = loss(fuzed_prob[0], grnd_truth)
    final_pred = None
    return loss_fused_seg, final_pred


class CustomTrainEpoch(smp_utils.train.TrainEpoch):
    def __init__(self, model, loss, metrics, optimizer,lr_scheduler, total_epochs, device='cpu', verbose=True,model_id=0,ds_version=-1):
        super().__init__(model, loss, metrics, optimizer, device=device, verbose=verbose)
        self.model_id = model_id
        self.ds_version = ds_version
        self.total_epochs = total_epochs  # Store the total number of epochs
        self.current_epoch = 0  # Initialize current_epoch to track progress
        self.lr_scheduler =  lr_scheduler
    def batch_update(self, x, grnd_truth):
        x, grnd_truth = x.to(self.device), grnd_truth.to(self.device)
        self.optimizer.zero_grad()

        # Get multiple outputs from the model
        logits = self.model(x)

        probs = logits2probs(logits)
        final_pred = probs2segs(probs)[0]
        loss = self.loss(probs[0].squeeze(), grnd_truth.squeeze().float())

        loss.backward()

        self.optimizer.step()
        return loss, final_pred

    def custom_run(self, dataloader, epoch):
        # Run the training epoch and update current_epoch
        self.current_epoch = epoch
        logs = super().run(dataloader)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(epoch + 1)  # Update scheduler based on epoch
        return logs


def run_train(model, train_loader, optimizer, loss, valid_loader, train_dataset, run, lr_scheduler, metrics, args):

    train_epoch = CustomTrainEpoch(
            model,
            loss=loss,
            metrics=metrics,
            optimizer=optimizer,
            lr_scheduler = lr_scheduler,
            total_epochs=args.epochs,
            device=args.device,
            verbose=True,
            model_id=args.model_id
        )

    best_iou_score = 0.0
    train_logs_list, valid_logs_list = [], []
    # Early stopping configuration

    no_improvement_counter = 0
    for epoch in range(0, args.epochs):

        # Perform training & validation
        print('\nEpoch: {}'.format(epoch))
        model.train()
        train_logs = train_epoch.custom_run(train_loader, epoch)
        train_logs_list.append(train_logs)

        torch.cuda.empty_cache()
        avg_iou = evaluate_model_validation(model, valid_loader, args)
        print_info_message(f"\n \n valid IoU (all image): {avg_iou:.4f}")


        if run is not None:
            run["train loss"].append(train_logs[loss._name])
            run["valid mIoU"].append(avg_iou)
        if best_iou_score < avg_iou:
            best_iou_score = avg_iou
            no_improvement_counter = 0  # Reset counter if improvement
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': loss,
                'best_iou_score(val)': best_iou_score
                # Add any other training metadata here
            }, args.path2save_model)
            print(args.path2save_model)
            print('Model saved!')
        else:
            no_improvement_counter += 1

        # Early stopping check
        if no_improvement_counter >= args.patience:
            print(f'Early stopping triggered after {epoch + 1} epochs. patience {no_improvement_counter}')
            break


        if args.visualize_segmentation_pred:
            # Visualize input image, ground truth, and prediction after each epoch
            image, gt_mask = train_dataset[0]  # get a sample from the validation dataset
            logits = model(image.unsqueeze(dim=0).cuda().float())

            pred_mask = predict(args.model_id, logits)

            visualize(
                ds_name=args.ds_names[args.ds_id],
                iou_sngl_img=np.round(avg_iou.cpu(),2),
                original_image= np.transpose(image , (1, 2, 0))/ 255,# do not change
                ground_truth_mask=colour_code_segmentation(gt_mask.int(), args.clss_rgb),
                predicted_mask=colour_code_segmentation(pred_mask.squeeze().cpu().numpy(), args.clss_rgb)
            )
