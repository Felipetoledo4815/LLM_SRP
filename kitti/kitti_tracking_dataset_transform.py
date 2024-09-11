import os
import json
import numpy as np
from numpy.core.defchararray import zfill

from dataset import config

def main():
    root_folder = config.kitti_tracking['root_folder'] if config.kitti_tracking['root_folder'].endswith(
            '/') else config.kitti_tracking['root_folder'] + '/'
    image_folders_path = root_folder + config.kitti_tracking['version'] + '/training/image_02/'
    all_image_folders = get_list_of_folder(image_folders_path)
    oxts_folder = root_folder + 'data_tracking_oxts/training/oxts/'
    label_folders = root_folder + 'data_tracking_label_2/training/label_02/'
    json_file_path = root_folder + 'converted_data.json'
    camera_calib_file_path = root_folder + 'data_tracking_calib/training/calib/'
    objects = []
    for image_folder in all_image_folders:
        path = image_folders_path + image_folder + '/'
        list_of_images = parse_image_data_from_folder(path)
        list_of_labels = get_list_of_labels_for_image_folder(label_folders + image_folder + '.txt')
        list_of_oxts = get_oxts_for_image(oxts_folder + image_folder + '.txt')
        camera_calib_info_p2 = get_camera_intrinsic(camera_calib_file_path + image_folder + '.txt')
        rotation_data_from_camera_calib = get_rotation_data(camera_calib_file_path + image_folder + '.txt')
        for idx, img in enumerate(list_of_images):
            file_object = {
                'image_folder': image_folder,
                'image_index': img,
                'image_path': path + img + '.png',
                'image_label_path': label_folders + image_folder + '.txt',
                'image_label': get_labels_for_image(list_of_labels, img),
                'image_oxts_path': oxts_folder + image_folder + '.txt',
                'image_oxts_data': list_of_oxts[idx],
                'oxts_data': oxts_data_parser(list_of_oxts[idx].split()),
                'camera_intrinsics': camera_calib_info_p2,
                'rotation': rotation_data_from_camera_calib
            }
            if len(file_object['image_label']) > 0:
                objects.append(file_object)
    print(len(objects))
    with open(json_file_path, 'w') as json_file:
        json.dump(objects, json_file, indent = 4)
    return
def get_rotation_data(path):
    with open(path,'r') as file:
        content = file.read()
        l = next(line for line in content.split('\n') if line.startswith('R_rect'))
        print(l)
        value = l.split(' ')[1:]
        value = [i for i in value if i]
    return value

def get_camera_intrinsic(camera_calib_file_path):
    # https://mmdetection3d.readthedocs.io/en/v0.17.3/datasets/kitti_det.html (P2: camera2 projection matrix after rectification, an 3x4 array)
    with open(camera_calib_file_path,'r') as file:
        content = file.read()
        l = next(line for line in content.split('\n') if line.startswith('P2'))
        _, value = l.split(':', 1)
        P2 = np.array([float(x) for x in value.split()]).reshape(3, 4)
        p2_matrix = P2[:3, :3]
    return p2_matrix.tolist()

def oxts_data_parser(oxts_data):
    return {
        'lat': oxts_data[0],
        'lon': oxts_data[1],
        'alt': oxts_data[2],
        'roll': oxts_data[3],
        'pitch': oxts_data[4],
        'yaw': oxts_data[5]
    }

def get_oxts_for_image(oxts_file_path):
    with open(oxts_file_path, 'r') as file:
        data = [line.strip() for line in file]
    return data

def get_labels_for_image(list_of_labels, img):
    img_label = []
    for label in list_of_labels:
        d = label.split()
        id = d[0].zfill(6)
        if id == img and d[2] != 'DontCare' and d[4] == '0': #d[4] ==  '0' is considering only fully visible objects
            print(id)
            obj = {
                'name': d[2],
                'bbox': {
                    'left': d[6],
                    'top': d[7],
                    'right': d[8],
                    'bottom': d[9]
                },
                'dimensions': {
                    'height': d[10],
                    'width': d[11],
                    'length': d[12]
                },
                'location': [d[13], d[14], d[15]],
                'rotating_y': d[16]
            }
            img_label.append(obj)
    return img_label

def get_list_of_labels_for_image_folder(image_label_path):
    with open(image_label_path, 'r') as file:
        label_data = [line.strip() for line in file]
    return label_data

def parse_image_data_from_folder(path):
    list_of_image_file_index = [os.path.splitext(f)[0] for f in os.listdir(path) if f.endswith('.png')]
    return list_of_image_file_index

def get_list_of_folder(path):
    items = os.listdir(path)
    folders = [item for item in items if os.path.isdir(os.path.join(path, item))]
    return folders

if __name__ == "__main__":
    main()