import urllib.request
from tqdm import tqdm
import click
import time
import os


def download_images(link_file_images, output_directory, image_type):

    print("\nDownloading", image_type)

    counter = 0

    with open(link_file_images, "r") as link_file:
        image_links = link_file.readlines()

    for image_link in tqdm(image_links, total=len(image_links)):
        image_path = os.path.join(
            output_directory + image_type, os.path.basename(image_link)
        ).strip()

        urllib.request.urlretrieve(image_link, image_path)

        counter += 1

    print("{} images downloaded to {}\n".format(counter, output_directory + image_type))


if __name__ == "__main__":
    dataset_name = "MassachusettsRoads"
    # os.path.join
    link_file_images = os.path.join(
        "..", "Data", "_Links", "MassachusettsRoads", "Images.txt"
    )
    link_file_targets = os.path.join(
        "..", "Data", "_Links", "MassachusettsRoads", "Targets.txt"
    )
    output_directory = os.path.join("..", "Data", dataset_name)

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
        os.mkdir(output_directory + "Images/")
        os.mkdir(output_directory + "Targets/")

    start_time = time.time()
    download_images(link_file_images, output_directory, "Images")
    download_images(link_file_targets, output_directory, "Targets")
    print("TOTAL TIME: {} minutes".format(round((time.time() - start_time) / 60, 2)))
