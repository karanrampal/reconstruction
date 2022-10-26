#!/usr/bin/env python
"""Convert pickle to json"""

import argparse
import json
import os
import pickle
import sys
from typing import Any, Dict, List, Union

sys.path.insert(0, "../bodyscan")

DataType = Union[float, List[List[float]], Dict[str, float]]


def args_parser() -> argparse.Namespace:
    """CLI argument parser"""
    parser = argparse.ArgumentParser(description="CLI for converting pickle to json")
    parser.add_argument(
        "-d",
        "--data_dir",
        type=str,
        default="data/data_10_19/calibration_2022_10_19_11_17_26_1666171046886500093",
        help="Data directory",
    )

    return parser.parse_args()


def read_pickle(path: str) -> Dict[str, Any]:
    """Read pickle file and return the data"""
    with open(path, "rb") as fin:
        data = pickle.load(fin)

    return data


def write_json(data: Dict[str, DataType], path: str) -> None:
    """Write data as a json file"""
    with open(path, "w", encoding="utf-8") as fout:
        json.dump(data, fout)


def main() -> None:
    """Main function"""
    args = args_parser()

    print("\nProcessing depth scales...")
    data = read_pickle(os.path.join(args.data_dir, "device_depth_scales.pkl"))
    write_json(data, os.path.join(args.data_dir, "device_depth_scales.json"))

    print("Processing transformations...")
    data = read_pickle(os.path.join(args.data_dir, "transformations.pkl"))
    out = {serial: mat.transformation_matrix.tolist() for serial, mat in data.items()}
    write_json(out, os.path.join(args.data_dir, "transformations.json"))

    print("Processing intrinsics...")
    intrin_files = {
        f.split("_")[0]: os.path.join(args.data_dir, f)
        for f in os.listdir(args.data_dir)
        if "intrinsics" in f
    }
    for serial, path in intrin_files.items():
        data = read_pickle(path)
        tmp = {
            "fx": data.fx,  # type: ignore[attr-defined]
            "fy": data.fy,  # type: ignore[attr-defined]
            "height": data.height,  # type: ignore[attr-defined]
            "width": data.width,  # type: ignore[attr-defined]
            "cx": data.ppx,  # type: ignore[attr-defined]
            "cy": data.ppy,  # type: ignore[attr-defined]
        }
        out = {serial: tmp}
        write_json(out, path[:-3] + "json")


if __name__ == "__main__":
    main()
