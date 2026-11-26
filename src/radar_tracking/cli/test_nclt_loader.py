from radar_tracking.data.nclt_loader import load_nclt_dataset


def main():
    data = load_nclt_dataset("real_world_pipeline/data/test.pt")

    print("Number of sequences:", len(data))

    obs, state = data[0]

    print("Obs shape:", obs.shape)
    print("State shape:", state.shape)


if __name__ == "__main__":
    main()
