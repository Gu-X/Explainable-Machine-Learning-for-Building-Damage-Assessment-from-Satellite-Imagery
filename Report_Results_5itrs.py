import os
import re
import numpy as np
from collections import defaultdict


def parse_test_results(file_path):
    """
    Parses a test results file and extracts the metrics into a structured format.
    """
    model_results = defaultdict(lambda: {"IoU": [], "Precision": [], "Accuracy": [], "F1-score": [], "Recall": []})

    with open(file_path, 'r') as file:
        content = file.read()
        # Split the content for each model entry
        model_entries = content.strip().split("\n\n")

        for entry in model_entries:
            # Extract model path
            model_match = re.search(r"model:\s*(.*)\s*on", entry)
            if model_match:
                model_path = model_match.group(1)
                # Extract metrics
                iou_match = re.search(r"IoU\s*:\s*(\d+\.\d+)", entry)
                precision_match = re.search(r"Precision\s*:\s*(\d+\.\d+)", entry)
                accuracy_match = re.search(r"Accuracy\s*:\s*(\d+\.\d+)", entry)
                f1_match = re.search(r"F1-score\s*:\s*(\d+\.\d+)", entry)
                recall_match = re.search(r"Recall\s*:\s*(\d+\.\d+)", entry)

                if all([iou_match, precision_match, accuracy_match, f1_match, recall_match]):
                    model_results[model_path]["IoU"].append(float(iou_match.group(1)))
                    model_results[model_path]["Precision"].append(float(precision_match.group(1)))
                    model_results[model_path]["Accuracy"].append(float(accuracy_match.group(1)))
                    model_results[model_path]["F1-score"].append(float(f1_match.group(1)))
                    model_results[model_path]["Recall"].append(float(recall_match.group(1)))

    return model_results


def aggregate_metrics_by_prefix(metrics):
    """
    Groups results by model prefix (excluding the iteration number) and calculates mean and standard deviation.
    """
    aggregated = defaultdict(lambda: {"IoU": [], "Precision": [], "Accuracy": [], "F1-score": [], "Recall": []})

    for model, metric_values in metrics.items():
        # Extract model prefix by removing the iteration number
        prefix = "_".join(model.split("_")[:-1])

        for metric, values in metric_values.items():
            aggregated[prefix][metric].extend(values)

    # Calculate mean and standard deviation for each metric
    final_results = {}
    for prefix, metric_values in aggregated.items():
        final_results[prefix] = {}
        for metric, values in metric_values.items():
            final_results[prefix][metric] = {
                "mean": np.mean(values),
                "std": np.std(values)
            }

    return final_results


# Directory containing the results files
results_dir = "Results_5Itrs"

# Parse and aggregate results
all_model_results = defaultdict(lambda: {"IoU": [], "Precision": [], "Accuracy": [], "F1-score": [], "Recall": []})
for i in range(5):  # Assuming 5 iterations
    file_name = f"paper_test_results_{i}.txt"
    file_path = os.path.join(results_dir, file_name)
    if os.path.exists(file_path):
        model_results = parse_test_results(file_path)
        for model, metrics in model_results.items():
            for metric, values in metrics.items():
                all_model_results[model][metric].extend(values)
    else:
        print(f"File {file_path} does not exist.")

# Aggregate metrics by prefix
aggregated_metrics = aggregate_metrics_by_prefix(all_model_results)

# Display results
print("Aggregated Metrics for Each Model Prefix (Mean ± Std):")
ds_names = ["Massa", "WHU", "Inria", "GBSS"]
ds_preprocessing = ["random cropping", "selected cropping"]
models_names = ["FPN", "Unet", "UNet++", "DeepLabV3+", "PSPNet", "UANet", "6 NO", "7 NO", "8 NO", "9 NO", "BSNet"]
for prefix, metrics in aggregated_metrics.items():
    match = re.search(r'bst_mdl_(\d+)_(\d+)_(\d+)', prefix)

    if match:
        numbers = match.groups()  # Extract the matched numbers as a tuple
        print("Model :", ds_names[int(numbers[1])], models_names[int(numbers[0])], ds_preprocessing[int(numbers[2])])
    else:
        print("No match found.")
    # for metric, values in metrics.items():
    #     print(f"  {metric}: {values['mean']:.2f} ± {values['std']:.2f}")
    latex_row = " & ".join(
        [f"{values['mean']:.2f} ± {values['std']:.2f}" for metric, values in metrics.items()]
    ) + " \\\\"
    print(latex_row)
