#!/usr/bin/env python
"""Download data from GCP bucket"""

import argparse
import json
import os
from datetime import datetime
from subprocess import check_call


def args_parser() -> argparse.Namespace:
    """CLI parser"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r", "--root", type=str, default="gs://measurement_bucket", help="Bucket name"
    )
    parser.add_argument(
        "-d", "--dir_path", type=str, help="Directory to download", required=True
    )
    parser.add_argument(
        "-o", "--out_path", type=str, default="./data", help="Output path"
    )

    return parser.parse_args()


def main() -> None:
    """Main function"""
    args = args_parser()

    print("\nMaking output folder...")
    name_ = "data_" + datetime.today().strftime("%Y-%m-%d")
    out_path = args.out_path + "/" + name_
    os.makedirs(out_path, exist_ok=True)

    print("Downloading images...")
    img_path = args.root + "/" + args.dir_path
    cmd = f"gsutil -m cp -r {img_path} {out_path}"
    check_call(cmd, shell=True)

    print("\nRead json file...")
    with open(
        os.path.join(out_path, args.dir_path, "metadata.json"), "r", encoding="utf-8"
    ) as fin:
        data = json.load(fin)
    calibs = data["calibration_blob_name"]

    print("Downloading calibrations...")
    calib_path = args.root + "/" + calibs
    cmd = f"gsutil -m cp -r {calib_path} {out_path}"
    check_call(cmd, shell=True)


if __name__ == "__main__":
    main()
