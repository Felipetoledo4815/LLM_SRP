import json
from dataset import config
import os
import numpy as np


def get_oxts_subpath(oxts_mapping):
    oxts_data = oxts_mapping.split()
    return oxts_data[1]+'/oxts/data/'+oxts_data[2]+'.txt'

def get_oxts_data_from_file(path):
    if os.path.isfile(path):
        with open(path, 'r') as file:
            content = file.read()
        return content
    else:
        return 0


def main():
    dataset_root_folder = config.kitti_3d_dataset['root_folder'] if config.kitti_3d_dataset['root_folder'].endswith(
            '/') else config.kitti_3d_dataset['root_folder'] + '/'
    image_folders_path = dataset_root_folder + config.kitti_3d_dataset['version'] + '/training/image_2/'

    oxts_folder = dataset_root_folder + 'oxts_raw_data/'
    oxts_mapping_file = dataset_root_folder + 'devkit_object/mapping/train_mapping.txt'
    oxts_mapping_index_file = dataset_root_folder + 'devkit_object/mapping/train_rand.txt'

    label_folders = dataset_root_folder + 'data_object_label_2/training/label_2/'

    json_file_path = dataset_root_folder + 'converted_data.json'

    camera_calib_file_path = dataset_root_folder + 'data_object_calib/training/calib/'

    list_of_images = parse_image_data_from_folder(image_folders_path)
    list_of_index_image_oxts_map = get_index_from_file(oxts_mapping_index_file)
    mapping_of_oxts_data_points = get_oxts_for_image(oxts_mapping_file)

    objects = []

    # should have 7481 images for training
    if len(list_of_images) == len(mapping_of_oxts_data_points) == len(list_of_index_image_oxts_map):
        print('We have exact amount of data' + str(len(list_of_images)))

    for index in range(len(list_of_images)):
        path = image_folders_path + list_of_images[index] + '.png'
        idx = list_of_index_image_oxts_map[index]
        oxts_mapping = mapping_of_oxts_data_points[int(idx)-1]
        oxts_file_path = oxts_folder + get_oxts_subpath(oxts_mapping)
        oxts_data = get_oxts_data_from_file(oxts_file_path)
        camera_calib_info_p2 = get_camera_intrinsic(camera_calib_file_path + list_of_images[index]+ '.txt')
        rotation_data_from_camera_calib = get_rotation_data(camera_calib_file_path + list_of_images[index]+ '.txt')
        list_of_labels = get_list_of_labels_for_image_folder(label_folders + list_of_images[index] + '.txt')

        if oxts_data == 0:
            continue
        file_obj = {
            'image_path': path,
            'image_serial': list_of_images[index],
            'index': idx,
            'oxts_mapping': oxts_mapping,
            'image_label_path': label_folders + list_of_images[index] + '.txt',
            'image_labels': get_labels_for_image(list_of_labels),
            'oxts_file_path': oxts_file_path,
            'oxts_data': oxts_data_parser(oxts_data.split()),
            'camera_intrinsics': camera_calib_info_p2,
            'rotation': rotation_data_from_camera_calib
        }
        objects.append(file_obj)
        # list_of_images = parse_image_data_from_folder(path)
        # list_of_labels = get_list_of_labels_for_image_folder(label_folders + image_folder + '.txt')
        # list_of_oxts = get_oxts_for_image(oxts_folder + image_folder + '.txt')
        # camera_calib_info_p2 = get_camera_intrinsic(camera_calib_file_path + image_folder + '.txt')
        # rotation_data_from_camera_calib = get_rotation_data(camera_calib_file_path + image_folder + '.txt')
        # for idx, img in enumerate(list_of_images):
        #     file_object = {
        #         'image_folder': image_folder,
        #         'image_index': img,
        #         'image_path': path + img + '.png',
        #         'image_label_path': label_folders + image_folder + '.txt',
        #         'image_label': get_labels_for_image(list_of_labels, img),
        #         'image_oxts_path': oxts_folder + image_folder + '.txt',
        #         'image_oxts_data': list_of_oxts[idx],
        #         'oxts_data': oxts_data_parser(list_of_oxts[idx].split()),
        #         'camera_intrinsics': camera_calib_info_p2,
        #         'rotation': rotation_data_from_camera_calib
        #     }
        #     if len(file_object['image_label']) > 0:
        #         objects.append(file_object)
    print(len(objects))
    with open(json_file_path, 'w') as json_file:
        json.dump(objects, json_file, indent = 4)
    return
def get_rotation_data(path):
    with open(path,'r') as file:
        content = file.read()
        l = next(line for line in content.split('\n') if line.startswith('R0_rect'))
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

def get_index_from_file(path):
    with open(path, 'r') as file:
        d = file.read()
    return d.split(',')

def get_labels_for_image(list_of_labels):
    img_label = []
    for label in list_of_labels:
        d = label.split()
        id = d[0].zfill(6)
        if d[0] != 'DontCare':
            print(id)
            obj = {
                'name': d[0],
                'truncated': d[1],
                'visibility': d[2],
                'bbox': {
                    'left': d[4],
                    'top': d[5],
                    'right': d[6],
                    'bottom': d[7]
                },
                'dimensions': {
                    'height': d[8],
                    'width': d[9],
                    'length': d[10]
                },
                'location': [d[11], d[12], d[13]],
                'rotating_y': d[14],
            }
            img_label.append(obj)
    return img_label

def get_list_of_labels_for_image_folder(image_label_path):
    with open(image_label_path, 'r') as file:
        label_data = [line.strip() for line in file]
    return label_data

def parse_image_data_from_folder(path):
    list_of_image_file_index = [os.path.splitext(f)[0] for f in os.listdir(path) if f.endswith('.png')]
    return sorted(list_of_image_file_index)

def get_list_of_folder(path):
    items = os.listdir(path)
    folders = [item for item in items if os.path.isdir(os.path.join(path, item))]
    return folders
if __name__ == "__main__":
    main()