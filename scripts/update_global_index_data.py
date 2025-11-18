#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Update global index data script.

This script fetches global index data from akshare and saves it to the
tests/test_data directory for use in unit tests.

Usage:
    python scripts/update_global_index_data.py [data_root]

If data_root is not specified, it defaults to 'tests/test_data'.
"""

import os
import sys
import time
from pathlib import Path

import akshare as ak
import pandas as pd

REQUEST_DELAY_SECONDS = 10
RETRY_DELAY_SECONDS = 10
MAX_RETRIES = 3


def get_data_root(default_path='tests/test_data', custom_path=None):
    """
    Determine the data root directory.

    Parameters
    ----------
    default_path : str, optional
        Default data root path (default: 'tests/test_data')
    custom_path : str, optional
        Custom data root path from command line argument

    Returns
    -------
    str
        Absolute path to the data root directory
    """
    if custom_path:
        data_root = custom_path
    else:
        data_root = default_path

    if not os.path.isabs(data_root):
        script_dir = Path(__file__).parent.resolve()
        project_root = script_dir.parent
        data_root = str(project_root / data_root)

    return data_root


def ensure_directory(path):
    """
    Ensure the directory exists, create if it doesn't.

    Parameters
    ----------
    path : str
        Path to the directory
    """
    Path(path).mkdir(parents=True, exist_ok=True)
    print(f"[OK] Data directory: {path}")


def update_global_index_data(data_root):
    """
    Fetch global index data from akshare and save to CSV files.

    Parameters
    ----------
    data_root : str
        Root directory to save data files
    """
    print(f"\n{'='*60}")
    print(f"Updating global index data")
    print(f"Data root: {data_root}")
    print(f"{'='*60}\n")

    ensure_directory(data_root)

    try:
        main_data_path = os.path.join(data_root, 'global_index_data.csv')

        if os.path.exists(main_data_path):
            print(f"[SKIP] Spot data already exists, loading from: {main_data_path}")
            spot_em_df = pd.read_csv(main_data_path)
            print(f"[OK] Loaded {len(spot_em_df)} global indices from local file")
        else:
            print("Fetching global index spot data...")
            spot_em_df = ak.index_global_spot_em()
            print(f"[OK] Retrieved {len(spot_em_df)} global indices")

            spot_em_df.to_csv(main_data_path, index=False)
            print(f"[OK] Saved main index data to: {main_data_path}")

        unique_names = spot_em_df['名称'].unique()
        print(f"\nFound {len(unique_names)} unique indices")

        print(f"\nFetching historical data for each index...")
        print(f"-" * 60)

        for i, name in enumerate(unique_names, 1):
            progress = f"[{i}/{len(unique_names)}]"
            safe_name = name.replace('/', '_')
            file_path = os.path.join(data_root, f'{safe_name}.csv')

            if os.path.exists(file_path):
                print(f"{progress} [SKIP] {name} already exists, skipping download")
                continue

            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    df = ak.index_global_hist_em(symbol=name)
                    df.to_csv(file_path, index=False)

                    print(f"{progress} [OK] {name}")
                    time.sleep(REQUEST_DELAY_SECONDS)
                    break

                except Exception as e:
                    if attempt < MAX_RETRIES:
                        print(
                            f"{progress} [WARN] Attempt {attempt} failed for {name}: {str(e)}. "
                            f"Retrying in {RETRY_DELAY_SECONDS}s..."
                        )
                        time.sleep(RETRY_DELAY_SECONDS)
                    else:
                        print(
                            f"{progress} [ERROR] Failed to fetch {name} after {MAX_RETRIES} attempts: {str(e)}"
                        )

        print(f"\n{'='*60}")
        print(f"[OK] Update complete!")
        print(f"[OK] Data saved to: {data_root}")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"\n[ERROR] Error updating global index data: {str(e)}")
        sys.exit(1)


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        custom_data_root = sys.argv[1]
    else:
        custom_data_root = None

    data_root = get_data_root(default_path='tests/test_data/index_data', custom_path=custom_data_root)
    update_global_index_data(data_root)


if __name__ == "__main__":
    main()
