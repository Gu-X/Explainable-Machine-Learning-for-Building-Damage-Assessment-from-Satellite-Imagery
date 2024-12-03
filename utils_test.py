from datetime import datetime

import torch
from utils_UI import print_info_message, print_error_message
from utils_evaluation import evaluate_model_testing


def run_test(model,tst_loader, args):


    model.load_state_dict(torch.load(args.test_model_path)['model_state_dict'])

    model.cuda()

    # Evaluate the model and store the results as a PyTorch tensor
    avg_iou, avg_precision, avg_accuracy, avg_recall, avg_f1 = evaluate_model_testing(model, tst_loader, args)

    print_info_message(
        f"\n \n Test results for model: {args.test_model_path} on all the image"
        f"\n IoU : {avg_iou.cpu().numpy()*100:.2f}"
        f"\n Precision: {avg_precision*100:.2f}"
        f"\n Accuracy: {avg_accuracy*100:.2f}"
        f"\n F1-score: {avg_f1*100:.2f}"
        f"\n Recall: {avg_recall*100:.2f}"
    )

    # Get the current date and time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Generate the info message
    message = (
        f"\n \n model: {args.test_model_path} on all the image"
        f", Date and Time: {current_time}"
        f"\n IoU : {avg_iou * 100:.2f}"
        f", Precision: {avg_precision * 100:.2f}"
        f", Accuracy: {avg_accuracy * 100:.2f}"
        f", F1-score: {avg_f1 * 100:.2f}"
        f", Recall: {avg_recall * 100:.2f}"
    )

    # Write the info message to the file
    write_info_message(message,args)
    print("Test results have been written to 'test_results_0.txt'")

# Define the function to write test results
def write_info_message(message,args):

    # Open the file in append mode; if it doesn't exist, create it
    with open(args.path2save_test_results, "a") as file:
        file.write(message + "\n")