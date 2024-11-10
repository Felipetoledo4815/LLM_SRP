from dataset.open_lane_v1_dataset import OpenLaneV1Dataset
from dataset.config import open_lane_v1


def main():
    d = OpenLaneV1Dataset(open_lane_v1)
    d.plot_data_point(19, "test.jpg")


if __name__ == "__main__":
    main()
