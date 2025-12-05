import numpy as np
from sklearn.cluster import KMeans
from skimage.segmentation import slic
from skimage.color import rgb2gray
from skimage.filters import sobel

def kmeans_partition(size, parts=256):
    H, W = size
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    coords = np.stack([yy.ravel(), xx.ravel()], axis=1).astype(np.float32)
    km = KMeans(n_clusters=parts, init="k-means++", n_init=10, random_state=0)
    labels = km.fit_predict(coords)
    centers = km.cluster_centers_
    order = np.lexsort((centers[:,1], centers[:,0]))
    remap = np.empty(parts, dtype=int)
    remap[order] = np.arange(parts)
    return remap[labels].reshape(H, W)


def build_ref_edge_image(images):
    images = iter(images)
    first = next(images)
    H, W = first.shape[:2]

    def to_edge(im):
        if im.ndim == 3 and im.shape[2] > 1:
            g = rgb2gray(im)
        else:
            g = im.astype(np.float32)
        g = (g - g.min()) / (g.ptp() + 1e-8)
        return sobel(g)

    accum = np.zeros((H, W), dtype=np.float32)
    count = 0

    # include first image
    accum += to_edge(first)
    count += 1

    for i, im in enumerate(images, start=1):
        accum += to_edge(im)
        count += 1

    avg_edge = accum / count
    ref_img = np.repeat(avg_edge[..., None], 3, axis=2).astype(np.float32)

    return ref_img


def superpixel_partition(ref_image, parts=256, compactness=10.0):
    H, W = ref_image.shape[:2]

    labels = slic(
        ref_image,
        n_segments=parts,
        compactness=compactness,
        start_label=0,
        enforce_connectivity=False,
    ).astype(np.int32)

    unique_labels = np.unique(labels)
    K = unique_labels.size

    # compute geometric centers of each region
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    centers = np.zeros((K, 2), dtype=np.float32)
    for i, lab in enumerate(unique_labels):
        mask = (labels == lab)
        centers[i, 0] = yy[mask].mean()
        centers[i, 1] = xx[mask].mean()

    # order regions top-to-bottom, left-to-right
    order = np.lexsort((centers[:, 1], centers[:, 0]))
    remap = np.empty(K, dtype=np.int32)
    remap[order] = np.arange(K)

    lut = np.zeros(unique_labels.max() + 1, dtype=np.int32)
    lut[unique_labels] = remap

    return lut[labels]


def square_atlas_grid(size=(128, 128), grid=(8, 8)):
    h, w = size
    rows, cols = grid
    atlas = np.zeros((h, w), dtype=np.int32)

    cell_h = h // rows
    cell_w = w // cols

    label = 1
    for i in range(rows):
        for j in range(cols):
            y0, y1 = i * cell_h, (i + 1) * cell_h
            x0, x1 = j * cell_w, (j + 1) * cell_w
            atlas[y0:y1, x0:x1] = label
            label += 1

    assert len(np.unique(atlas)) == rows * cols, f"Got {len(np.unique(atlas))} labels"
    assert (atlas > 0).all(), "Background zeros detected"

    return atlas