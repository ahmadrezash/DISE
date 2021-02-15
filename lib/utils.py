from lib.settings import CLUSTER_IMAGE_PATH


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 200, 200)
    return x


import matplotlib.pyplot as plt


def show_image(images, cols: int = 5):
    plt.figure(figsize=(20, 10))
    columns = cols or 10
    for i, image in enumerate(images):
        plt.subplot(len(images) / columns + 1, columns, i + 1)
        plt.imshow(image, )
        plt.axis("off")
    plt.subplots_adjust(wspace=.05, hspace=.05)
    plt.show()


def save_image(images, cols: int = 5, name: str = "cluster"):
    plt.figure(figsize=(20, 10))
    columns = cols or 10
    for i, image in enumerate(images):
        plt.subplot(len(images) / columns + 1, columns, i + 1)
        plt.imshow(image, )
        plt.axis("off")
    plt.subplots_adjust(wspace=.05, hspace=.05)
    plt.savefig(f'{CLUSTER_IMAGE_PATH}/cluster-{name}.jpg')
