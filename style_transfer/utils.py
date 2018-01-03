import PIL.Image
import numpy as np
import matplotlib.pyplot as plt


def load_image(filename, shape=None, max_size=None):
    image = PIL.Image.open(filename)

    if max_size is not None:
        # Calculate the appropriate rescale-factor for
        # ensuring a max height and width, while keeping
        # the proportion between them.
        factor = float(max_size) / np.max(image.size)

        # Scale the image's height and width.
        size = np.array(image.size) * factor

        # The size is now floating-point because it was scaled.
        # But PIL requires the size to be integers.
        size = size.astype(int)

        # Resize the image.
        image = image.resize(size, PIL.Image.LANCZOS)  # PIL.Image.LANCZOS is one of resampling filter

    if shape is not None:
        image = image.resize(shape, PIL.Image.LANCZOS)  # PIL.Image.LANCZOS is one of resampling filter

    # Convert to numpy floating-point array.
    return np.float32(image)


def save_image(image, path):
    plt.imsave(path, image.astype(np.uint8))
