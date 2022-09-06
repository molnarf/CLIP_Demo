import os
import zipfile
import matplotlib.pyplot as plt
from PIL import Image
from fastapi import File
import numpy as np
from pathlib import Path
import shutil


def unzip_folder(folder_zip: File, output_path: str = "unzipped"):
    Path("input").mkdir(parents=True, exist_ok=True)
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    Path(output_path).mkdir(parents=True, exist_ok=True)


    with open("input/zip_file.zip", 'wb') as new_file:
        new_file.write(folder_zip)

        with zipfile.ZipFile("input/zip_file.zip", "r") as zip_file:
            zip_file.extractall(output_path)

def load_images(folder) -> list:
    images = []
    names = []

    #print(folder.file)

    for r, d, f in os.walk(folder):     # r: root, d: directory, f: files
        for filename in f:
            if filename.endswith(".png") or filename.endswith(".PNG") or filename.endswith(".jpg"):
                name = os.path.splitext(filename)[0]
                image = Image.open(os.path.join(r, filename)).convert("RGB")
                images.append(image)
                names.append(name)


    #for filename in [filename for filename in os.listdir(folder_path) if filename.endswith(".png") or filename.endswith(".PNG") or filename.endswith(".jpg")]:
    return images, names

def load_labels_from_file(path: str) -> list:
    with open(path) as f:
        lines = f.read().splitlines()
    return lines


def plot_best_predictions(images, labels, top_probs, top_labels):
    plot_path = "results.png"
    fig = plt.figure(figsize=(10, 7))
    num_rows = int(len(images) / 2) + 1

    for i, image in enumerate(images):
        plt.subplot(num_rows, 4, 2 * i + 1)
        plt.imshow(image)
        plt.axis("off")

        plt.subplot(num_rows, 4, 2 * i + 2)
        y = np.arange(top_probs.shape[-1])
        plt.grid()
        plt.barh(y, top_probs[i])
        plt.gca().invert_yaxis()
        plt.gca().set_axisbelow(True)
        plt.yticks(y, [labels[index] for index in top_labels[i].numpy()])
        plt.xlabel("probability")

    plt.subplots_adjust(wspace=0.5)
    plt.tight_layout()
    #plt.show() 

    fig.savefig(plot_path, dpi=80)
    return plot_path


