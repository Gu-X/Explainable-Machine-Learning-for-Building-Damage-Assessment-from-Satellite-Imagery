import cv2
import random
import warnings
import numpy as np
warnings.filterwarnings("ignore")
from utils_visualize import visualize_roi
from scipy.ndimage import label, center_of_mass, sum as ndi_sum


def crop_region(image, mask, centroid,crop_size=256,visualize_crops=False):
    y, x = int(centroid[0]), int(centroid[1])

    # Calculate crop boundaries
    half_crop = crop_size // 2
    start_x = max(x - half_crop, 0)
    start_y = max(y - half_crop, 0)
    end_x = min(x + half_crop, image.shape[1])
    end_y = min(y + half_crop, image.shape[0])

    # Crop the image
    cropped_image = image[start_y:end_y, start_x:end_x]
    cropped_mask = mask[start_y:end_y, start_x:end_x]

    centroid = centroid[0] - start_y,centroid[1] - start_x

    # visualize the cropped roi
    if visualize_crops:
        visualize_roi(cropped_image, cropped_mask, centroid)

    return cropped_image, cropped_mask


def single_image_meta_data(centroids,image,mask,visualize_crops):
    meta_data_list = []
    crops_list = []

    # now crop the image and visualize each crop 256X256
    for centroid in centroids:
        roi_meta_data = {"roi_total_area": 0, "roi_num_classes": 0}
        roi_img,roi_mask = crop_region(image, mask, centroid, crop_size=256,visualize_crops=visualize_crops)
        # get all objects in this roi
        labeled_mask, num_objects_labels = label(roi_mask)

        num_classes_in_roi = np.unique(roi_mask)
        roi_meta_data["roi_num_classes"] = len(num_classes_in_roi)


        objects_areas = []
        for obj_label in range(num_objects_labels):
             objects_areas.append(np.sum(labeled_mask == obj_label))

        roi_meta_data["roi_area_mean"] = np.mean(objects_areas)
        roi_meta_data["roi_area_min"] = np.min(objects_areas)
        roi_meta_data["roi_area_max"] = np.max(objects_areas)
        meta_data_list.append(roi_meta_data)

        crops_list.append((roi_img,roi_mask))
        if crops_list is None:
            print(1)
    return meta_data_list,crops_list

def centre_pad_image(image,truth_mask):
    # Assuming crop_image is your input image with shape (256, 128, 3)

    # Desired size
    desired_height = 256
    desired_width = 256

    # Calculate padding sizes
    padding_top = (desired_height - image.shape[0]) // 2
    padding_bottom = desired_height - image.shape[0] - padding_top
    padding_left = (desired_width - image.shape[1]) // 2
    padding_right = desired_width - image.shape[1] - padding_left

    # Pad the image
    padded_image = cv2.copyMakeBorder(image, padding_top, padding_bottom, padding_left, padding_right,
                                      cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # Pad the image
    padded_mask = cv2.copyMakeBorder(truth_mask,padding_top, padding_bottom,padding_left, padding_right,
                                     cv2.BORDER_CONSTANT, value=[0])
    return padded_image,padded_mask


def one_image(image_path,image, mask, min_pixel_count=-1,max_pixel_count =-1, visualize_image=False,visualize_crops=False,
              ds_names=None,ds_id=-1,clss_rgb=None,analysisMode=False, MAX_NUM_CROPS=1000000):

    # get all objects in the big image the mask should be only HxW for all datasets at this point!
    objects_masks, num_objects = label(mask)

    # Calculate the size of each label
    objects_areas = ndi_sum(mask, objects_masks, range(1, num_objects + 1))

    # Calculate the centroids of each label
    centroids = center_of_mass(mask, objects_masks, range(1, num_objects + 1))

    # Create a new labeled image, only keeping regions within the size criteria
    filtered_centroids = []
    filtered_aras = []
    for i, size in enumerate(objects_areas):
        if min_pixel_count <= size <= max_pixel_count:
            filtered_centroids.append(centroids[i])
            filtered_aras.append(objects_areas[i])

    centroids = filtered_centroids

    # Define a margin to ensure 256x256 crop fits within the image
    margin = 128

    # Image dimensions
    height, width = mask.shape

    # Filter out centroids near the border
    centroids = [
        centroid for centroid in centroids
        if (centroid[0] >= margin and centroid[0] <= height - margin and
            centroid[1] >= margin and centroid[1] <= width - margin)]

    MAX_NUM_CROPS = min(len(centroids),MAX_NUM_CROPS)
    slcted_centroids = random.sample(centroids, MAX_NUM_CROPS)
    meta_data_list,crops_list = single_image_meta_data(slcted_centroids, image, mask, visualize_crops)

    # image has no objects or suitable objects
    if len(centroids) == 0 or len(meta_data_list)==0:
        if analysisMode:
            return None
        else:
            # print("utils_analysis_L147")
            return [],None,None

    if len(meta_data_list) ==0:
        print(1)

    image_meta_data = {
        "image_path": image_path,
        "meta_data": meta_data_list
    }

    if analysisMode==True:
       return image_meta_data

    else:#train mode
        counter = 0
        while(counter<=30):
            random_crop_id = random.randint(0, len(crops_list) - 1)
            crop_image = crops_list[random_crop_id][0]
            crop_mask = crops_list[random_crop_id][1]
            if crop_image.shape[0] == 256 and crop_image.shape[1] == 256:
                return image_meta_data,  crop_image, crop_mask
            else:
                counter = counter + 1
                # try next random crop
                # print("looking for a another suitable crop!")
        print("I had to pad the crop!")
        crop_image, crop_mask = centre_pad_image(crop_image,crop_mask)
        return image_meta_data,crop_image, crop_mask