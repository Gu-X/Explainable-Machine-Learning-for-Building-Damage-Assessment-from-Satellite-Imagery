import cv2
import matplotlib.pyplot as plt
import warnings
import numpy as np
from matplotlib import patches

warnings.filterwarnings("ignore")

# helper function for data visualization
def visualize(iou_sngl_img=-1,ds_name="None",**images):
    """
    Plot images in one row
    """
    n_images = len(images)

    plt.figure(figsize=(20, 8))
    if iou_sngl_img!=-1:
        plt.suptitle(f"Dataset: {ds_name}, Random Image IoU: {iou_sngl_img}%", fontsize=24)
    else:
        plt.suptitle(f"Dataset: {ds_name}", fontsize=24)


    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([])
        plt.yticks([])
        # get title from the parameter names
        plt.title(name.replace('_', ' ').title(), fontsize=20)
        plt.imshow(image)

    plt.show()


def draw_x_on_centroids(image, centroids, size=5, color=(255, 0, 255), thickness=2):
    # Ensure the image is a NumPy array of type uint8
    if not isinstance(image, np.ndarray):
        image = np.array(image, dtype=np.uint8)
    else:
        image = image.astype(np.uint8)

    # If the image is in RGB format, convert to BGR for OpenCV (optional, depends on your image format)
    if image.shape[-1] == 3:  # Checking if it's a 3-channel image
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw the 'X' at each centroid
    for centroid in centroids:
        y, x = int(centroid[0]), int(centroid[1])
        cv2.line(image, (x - size, y - size), (x + size, y + size), color, thickness)
        cv2.line(image, (x - size, y + size), (x + size, y - size), color, thickness)

    return image


# def visualize_entire_image(image_path,centroids,ds_name,**images):
#     n_images = len(images)
#     plt.figure(figsize=(10, 10))
#
#
#     plt.suptitle(f"Dataset: {image_path}", fontsize=24)
#
#     for idx, (name, image) in enumerate(images.items()):
#         plt.subplot(1, n_images, idx + 1)
#         plt.xticks([])
#         plt.yticks([])
#         # get title from the parameter names
#         plt.title(name.replace('_', ' ').title(), fontsize=20)
#
#         # Draw centroids on the image
#         image_with_x = draw_x_on_centroids(image, centroids)
#
#         # Convert back to RGB for matplotlib if needed
#         image_with_x = cv2.cvtColor(image_with_x, cv2.COLOR_BGR2RGB)
#
#         plt.imshow(image_with_x)
#
#     plt.show()





def visualize_roi(cropped_image,cropped_mask,centroid):
    # Plot the cropped image and mask side by side
    plt.figure(figsize=(8, 4))  # Adjust figure size as needed
    plt.subplot(1, 2, 1)
    plt.imshow(cropped_image)
    plt.title("Cropped Image")
    plt.xticks([]), plt.yticks([])

    plt.subplot(1, 2, 2)

    colour_codes = np.array([[0, 0, 0], [0, 255, 0], [255, 200, 50], [255, 0, 0], [0, 0, 255], [255, 255, 255]])
    # cropped_mask[cropped_mask == 255] = 5
    # colour_codes = np.array(label_values)

    cropped_mask = colour_codes[cropped_mask]

    cropped_mask = draw_x_on_centroids(cropped_mask, [centroid])

    plt.imshow(cropped_mask)  # Use 'gray' cmap for mask
    plt.title("Cropped Mask")
    plt.xticks([]), plt.yticks([])

    # Create a legend with colored boxes
    legend_elements = [
        patches.Patch(color='green', label='No Damage'),
        patches.Patch(color='orange', label='Minor Damage'),
        patches.Patch(color='red', label='Major Damage'),
        patches.Patch(color='blue', label='Destroyed'),
        patches.Patch(color='white', label='Un-Classified')
    ]

    plt.figtext(0.5, 0.01, '', ha='center', va='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    # Add legend elements manually to the plot
    plt.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.55), ncol=2, fontsize=12)

    plt.show()