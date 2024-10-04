from dataset.kitti_dataset import KittiDataset
from dataset.nuscenes_dataset import NuscenesDataset
from dataset.utils.plot import ScenePlot
from dataset import config
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_entity_corners_on_3d_plain(entities):
    list_corners = []
    for entity in entities:
        list_corners.append(entity.corners())

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for corner in list_corners:
        x = corner[0]
        y = corner[1]
        z = corner[2]

        # Plot the points
        ax.scatter(x, y, z, c='r', marker='o')
        # Plot the points
        # Connect the points with lines
        for j in range(len(x)):
            ax.text(x[j], y[j], z[j], str(j), fontsize=19)

        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], 'b-', alpha=0.5)

    # Set labels
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # Set equal aspect ratio
    max_range = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]).ptp().max() / 2.0
    mid_x = np.mean(ax.get_xlim())
    mid_y = np.mean(ax.get_ylim())
    mid_z = np.mean(ax.get_zlim())
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Add a color bar

    # Show the plot
    plt.show()

def main():
    kitti = KittiDataset(config.kitti_3d_dataset)
    idx = 3201 #4631 265 3201
    ego_vehicle = kitti.get_ego_vehicle(idx)
    entities = kitti.get_entities(idx)
    image_path = kitti.get_image(idx)
    print('\n'.join(map(str, kitti.get_sg_triplets(idx))))
    bb_triplets = kitti.get_bb_triplets(idx)
    print('BB Triplets')
    print(bb_triplets)

    scene_plot = ScenePlot(field_of_view=kitti.field_of_view)
    scene_plot.render_scene(ego_vehicle, entities, image_path)

    scene_plot.plot_2d_bounding_boxes([entities[1]], image_path)

    # print("we are here")
    # scene_plot.plot_2d_bounding_boxes_from_corners([bb_triplets[0][0]], image_path)
    # scene_plot.plot_2d_bounding_boxes_from_corners([bb_triplets[0][0]],
    #                                                image_path,
    #                                                 entity_types=[bb_triplets[0][1][0][0]])


if __name__ == "__main__":
    main()
