import json
from xml.etree.ElementTree import indent

from dataset import config
import os
def main():
    root_folder = config.kitti['root_folder'] if config.kitti['root_folder'].endswith(
            '/') else config.kitti['root_folder'] + '/'
    image_folder = root_folder + config.kitti['version'] + '/data/'
    oxts_folder = root_folder + 'oxts/data/'
    list_of_image_file_index = [os.path.splitext(f)[0] for f in os.listdir(image_folder) if f.endswith('.png')]
    # now read data from oxts
    json_file_path = root_folder + 'converted_data.json'
    objects = []
    for idx in list_of_image_file_index:
        oxts_file_path = oxts_folder + idx + '.txt'
        if os.path.exists(oxts_file_path):
            with open(oxts_file_path, 'r') as file:
                content = file.read()
                elements = content.split()
                # data structure is given in the oxts dataformat.txt file
                file_object = {
                    'image_index': idx,
                    'image_path': image_folder + idx + '.png',
                    'oxts_file_path': oxts_file_path,
                    'oxts_content': content,
                    'lat': elements[0],
                    'lon': elements[1],
                    'alt': elements[2],
                    'roll': elements[3],
                    'pitch': elements[4],
                    'yaw': elements[5]
                }
                # need to parse contents and get our desired datas
                objects.append(file_object)
        else:
            print(idx + 'file is missing')
    if len(objects) == len(list_of_image_file_index):
        print("we have all values")
    with open(json_file_path, 'w') as json_file:
        json.dump(objects, json_file, indent = 4)
    return
if __name__ == "__main__":
    main()