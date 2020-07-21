import cv2
import numpy as np


def plot_tiles(im_h, im_w, chips):
    img = np.ones((im_h, im_w, 3), dtype="uint8")*255
    color_idx = 0
    colors = [(255, 0, 0), (0, 0, 255)]
    for chip in chips:
        i, j = chip['i'], chip['j']
        x1, y1, x2, y2 = chip['x1'], chip['y1'],chip['x2'], chip['y2']
        color = colors[(i+j)%len(colors)]
        color_idx+=1
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)
        img = cv2.addWeighted(overlay, .4, img, 1 - .4, 0)
    scale_percent = 10  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized

def tile_image(im_w, im_h, c_w, c_h, overlap_w, overlap_h):
    m = im_w // (c_w - overlap_w)
    n = im_h // (c_h - overlap_h)
    m_remainder = im_w - (m*(c_w))
    n_remainder = im_h - (n*(c_h))

    tiles = []
    for i in range(m+1):
        for j in range(n+1):
            x1 = i*(c_w - overlap_w) if i !=m else im_w - c_w
            x2 = x1 + c_w
            y1 = j*(c_h - overlap_h) if j !=n else im_h - c_h
            y2 = y1 + c_h
            tiles.append({'x1': x1, 'x2':x2, 'y1':y1, 'y2':y2, 'i':i, 'j':j})
    return tiles

def plot_chips(image,tiles):
    w = image['w']
    h = image['h']
    tile_demo = plot_tiles(h,w,tiles)
    return tile_demo

def sample_chips(image, labels, chip_dim, chip_overlap_w, chip_overlap_h):
    w = image['w']
    h = image['h']
    tiles = tile_image(w, h, chip_dim, chip_dim, chip_overlap_w, chip_overlap_h)
    return tiles

def percent_on(chip, label):
    dx = min(chip['x2'], label['x2']) - max(chip['x1'], label['x1'])
    dy = min(chip['y2'], label['y2']) - max(chip['y1'], label['y1'])
    if (dx < 0) and (dy < 0):
        return None
    intersected_area = dx * dy
    label_area = (label['x2'] - label['x1']) * (label['y2'] - label['y1'])
    return intersected_area / label_area
