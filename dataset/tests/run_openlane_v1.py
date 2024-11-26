from dataset.open_lane_v1_dataset import OpenLaneV1Dataset
from dataset.config import open_lane_v1


def main():
    d = OpenLaneV1Dataset(open_lane_v1)
    d.plot_data_point(146, "test.jpg")
    print(d.get_sg_triplets(146))


if __name__ == "__main__":
    main()
