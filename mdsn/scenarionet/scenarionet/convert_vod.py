desc = "Build database from VOD scenarios"

prediction_split = ["train", "train_val", "val", "test"]
scene_split = ["v1.0-trainval", "v1.0-test"]

split_to_scene = {
    "train": "v1.0-trainval",
    "train_val": "v1.0-trainval",
    "val": "v1.0-trainval",
    "test": "v1.0-test",
}

if __name__ == "__main__":
    import pkg_resources  # for suppress warning
    import argparse
    import os.path
    from functools import partial
    from scenarionet import SCENARIONET_DATASET_PATH
    from scenarionet.converter.vod.utils import (
        convert_vod_scenario,
        get_vod_scenarios,
        get_vod_prediction_split,
    )
    from scenarionet.converter.utils import write_to_directory

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--database_path",
        "-d",
        default=os.path.join(SCENARIONET_DATASET_PATH, "vod"),
        help="directory, The path to place the data",
    )
    parser.add_argument(
        "--dataset_name",
        "-n",
        default="vod",
        help="Dataset name, will be used to generate scenario files",
    )
    parser.add_argument(
        "--split",
        default="v1.0-trainval",
        choices=scene_split + prediction_split,
        help="Which splits of VOD data should be used. If set to {}, it will convert the full log into scenarios"
        " with 20 second episode length. If set to {}, it will convert segments used for VOD prediction"
        " challenge to scenarios, resulting in more converted scenarios. Generally, you should choose this "
        " parameter from {} to get complete scenarios for planning unless you want to use the converted scenario "
        " files for prediction task.".format(scene_split, prediction_split, scene_split),
    )
    parser.add_argument("--dataroot", default="/data/sets/vod", help="The path of vod data")
    parser.add_argument("--map_radius", default=500, type=float, help="The size of map")
    parser.add_argument(
        "--future",
        default=3,
        type=float,
        help="3 seconds by default. How many future seconds to predict. Only "
        "available if split is chosen from {}".format(prediction_split),
    )
    parser.add_argument(
        "--past",
        default=0.5,
        type=float,
        help="0.5 seconds by default. How many past seconds are used for prediction."
        " Only available if split is chosen from {}".format(prediction_split),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If the database_path exists, whether to overwrite it",
    )
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers to use")
    args = parser.parse_args()

    overwrite = args.overwrite
    dataset_name = args.dataset_name
    output_path = args.database_path
    version = args.split

    if version in scene_split:
        scenarios, vods = get_vod_scenarios(args.dataroot, version, args.num_workers)
    else:
        scenarios, vods = get_vod_prediction_split(args.dataroot, version, args.past, args.future, args.num_workers)
    write_to_directory(
        convert_func=convert_vod_scenario,
        scenarios=scenarios,
        output_path=output_path,
        dataset_version=version,
        dataset_name=dataset_name,
        overwrite=overwrite,
        num_workers=args.num_workers,
        vodelft=vods,
        past=[args.past for _ in range(args.num_workers)],
        future=[args.future for _ in range(args.num_workers)],
        prediction=[version in prediction_split for _ in range(args.num_workers)],
        map_radius=[args.map_radius for _ in range(args.num_workers)],
    )
