import json

def darknet_format(json_local_path, im_w, im_h, species_map):
    with open(json_local_path) as f:
        labels = json.load(f)

    lines = []
    for label in labels:

        x1 = label['x1']
        x2 = label['x2']
        y1 = label['y1']
        y2 = label['y2']
        x1 = min(0, x1)
        x2 = max(im_w, x2)
        y1 = min(0, y1)
        y2 = max(im_h, y2)
        species_name = label['species']

        class_id = species_map[species_name]
        x_center = (x1 + (x2 - x1) / 2.0) / im_w
        y_center = (y1 + (y2 - y1) / 2.0) / im_h
        w = (x2 - x1) / im_w
        h = (y2 - y1) / im_h
        lines.append("%d %f %f %f %f" % (class_id, x_center, y_center, w, h))

    label_fp = json_local_path.replace('.json', '.txt')
    with open(label_fp, 'w') as text_file:
        for line in lines:
            text_file.write(line + '\n')

    return label_fp