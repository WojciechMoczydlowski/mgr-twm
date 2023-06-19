import numpy as np
import cv2
from tqdm import tqdm
import os
import math
import time


def train_test_split(images_path, masks_path, test_split=0.3):

    image_filenames = [filename for filename in os.walk(images_path)][0][2]
    test_set_size = int(test_split * len(image_filenames))

    root_path = os.path.dirname(os.path.dirname(images_path))
    train_dir = os.path.join(root_path, "Train")
    test_dir = os.path.join(root_path, "Test")

    if not os.path.exists(train_dir):
        print("CREATING:", train_dir)
        os.makedirs(os.path.join(train_dir, "Images", "samples"))
        os.makedirs(os.path.join(train_dir, "Masks", "samples"))

    if not os.path.exists(test_dir):
        print("CREATING:", test_dir)
        os.makedirs(os.path.join(test_dir, "Images", "samples"))
        os.makedirs(os.path.join(test_dir, "Masks", "samples"))

    train_image_dir = os.path.join(train_dir, "Images", "samples")
    train_mask_dir = os.path.join(train_dir, "Masks", "samples")
    test_image_dir = os.path.join(test_dir, "Images", "samples")
    test_mask_dir = os.path.join(test_dir, "Masks", "samples")

    for n, filename in enumerate(image_filenames):
        if n < test_set_size:
            os.rename(
                os.path.join(images_path, filename),
                os.path.join(test_image_dir, filename),
            )
            os.rename(
                os.path.join(masks_path, filename),
                os.path.join(test_mask_dir, filename),
            )
        else:
            os.rename(
                os.path.join(images_path, filename),
                os.path.join(train_image_dir, filename),
            )
            os.rename(
                os.path.join(masks_path, filename),
                os.path.join(train_mask_dir, filename),
            )

    print(
        "Train-Test-Split COMPLETED.\nNUMBER OF IMAGES IN TRAIN SET:{}\nNUMBER OF IMAGES IN TEST SET: {}".format(
            len(image_filenames) - test_set_size, test_set_size
        )
    )
    print("\nTrain Directory:", train_dir)
    print("Test Directory:", test_dir)


def crop_and_save(
    images_path, masks_path, new_images_path, new_masks_path, img_width, img_height
):
    print("Building Dataset.")

    num_skipped = 0
    start_time = time.time()
    files = next(os.walk(images_path))[2]
    print("Total number of files =", len(files))

    for image_file in tqdm(files, total=len(files)):
        image_path = os.path.join(images_path, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        if image is None:
            print("Image skipped", image_path)
            continue

        mask_path = os.path.join(masks_path, image_file[:-1])
        mask = cv2.imread(mask_path, 0)

        num_splits = math.floor(
            (image.shape[0] * image.shape[1]) / (img_width * img_height)
        )
        counter = 0

        for r in range(0, image.shape[0], img_height):
            for c in range(0, image.shape[1], img_width):
                counter += 1
                blank_image = np.zeros((img_height, img_width, 3), dtype=int)
                blank_mask = np.zeros((img_height, img_width), dtype=int)

                new_image_path = os.path.join(
                    new_images_path, str(counter) + "_" + image_file
                )
                new_mask_path = os.path.join(
                    new_masks_path, str(counter) + "_" + image_file
                )

                new_image = np.array(image[r : r + img_height, c : c + img_width, :])
                new_mask = np.array(mask[r : r + img_height, c : c + img_width])

                blank_image[: new_image.shape[0], : new_image.shape[1], :] += new_image
                blank_mask[: new_image.shape[0], : new_image.shape[1]] += new_mask

                blank_mask[blank_mask > 1] = 255

                # Skip any Image that is more than 95% empty.
                if np.any(blank_mask):
                    num_black_pixels, num_white_pixels = np.unique(
                        blank_mask, return_counts=True
                    )[1]

                    if num_white_pixels / num_black_pixels < 0.05:
                        num_skipped += 1
                        continue

                    if blank_image.min() == 255 and blank_image.max() == 255:
                        print("Skipped image", image_path)
                        continue

                    blank_image = blank_image.astype(np.uint8)
                    blank_mask = blank_mask.astype(np.uint8)

                    cv2.imwrite(new_image_path, blank_image)
                    cv2.imwrite(new_mask_path, blank_mask)

    print(
        "EXPORT COMPLETE: {} seconds.\nImages exported to {}\nMasks exported to{}".format(
            round((time.time() - start_time), 2), new_images_path, new_masks_path
        )
    )
    print("\n{} Images were skipped.".format(num_skipped))


if __name__ == "__main__":
    root_data_path = os.path.join("Data", "MassachusettsRoads")
    new_root_data_path = os.path.join("Data", "BuildingsDataSet")
    test_to_train_ratio = 0.3
    img_width = img_height = 256
    num_channels = 3

    # Path Information
    images_path = os.path.join(root_data_path, "Images")
    masks_path = os.path.join(root_data_path, "Targets")
    new_images_path = os.path.join(new_root_data_path, "Images")
    new_masks_path = os.path.join(new_root_data_path, "Masks")

    for path in [new_root_data_path, new_images_path, new_masks_path]:
        if not os.path.exists(path):
            os.mkdir(path)
            print("DIRECTORY CREATED: {}".format(path))
        else:
            print("DIRECTORY ALREADY EXISTS: {}".format(path))

    crop_and_save(
        images_path, masks_path, new_images_path, new_masks_path, img_width, img_height
    )
    train_test_split(new_images_path, new_masks_path, test_to_train_ratio)
