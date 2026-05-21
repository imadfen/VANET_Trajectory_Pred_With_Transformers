import os
from multiprocessing import Pool, cpu_count
import re
import logging

import numpy as np
import pandas as pd


logger = logging.getLogger("__main__")


class Normalizer(object):
    """
    Normalizes dataframe across ALL contained rows (time steps). Different from per-sample normalization.
    """

    def __init__(self, norm_type, mean=None, std=None, min_val=None, max_val=None):
        """
        Args:
            norm_type: choose from:
                "standardization", "minmax": normalizes dataframe across ALL contained rows (time steps)
                "per_sample_std", "per_sample_minmax": normalizes each sample separately (i.e. across only its own rows)
            mean, std, min_val, max_val: optional (num_feat,) Series of pre-computed values
        """

        self.norm_type = norm_type
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, df):
        """
        Args:
            df: input dataframe
        Returns:
            df: normalized dataframe
        """
        if self.norm_type == "standardization":
            if self.mean is None:
                self.mean = df.mean()
                self.std = df.std()
            return (df - self.mean) / (self.std + np.finfo(float).eps)

        elif self.norm_type == "minmax":
            if self.max_val is None:
                self.max_val = df.max()
                self.min_val = df.min()
            return (df - self.min_val) / (
                self.max_val - self.min_val + np.finfo(float).eps
            )

        elif self.norm_type == "per_sample_std":
            grouped = df.groupby(by=df.index)
            return (df - grouped.transform("mean")) / grouped.transform("std")

        elif self.norm_type == "per_sample_minmax":
            grouped = df.groupby(by=df.index)
            min_vals = grouped.transform("min")
            return (df - min_vals) / (
                grouped.transform("max") - min_vals + np.finfo(float).eps
            )

        else:
            raise (NameError(f'Normalize method "{self.norm_type}" not implemented'))

    def inverse_normalize(self, df):
        if self.norm_type == "standardization":
            return df * self.std + self.mean
        elif self.norm_type == "minmax":
            return df * (self.max_val - self.min_val) + self.min_val
        elif self.norm_type == "per_sample_std":
            grouped = df.groupby(by=df.index)
            return df * grouped.transform("std") + grouped.transform("mean")
        elif self.norm_type == "per_sample_minmax":
            grouped = df.groupby(by=df.index)
            min_vals = grouped.transform("min")
            max_vals = grouped.transform("max")
            return df * (max_vals - min_vals) + min_vals
        else:
            raise NameError(
                f'Inverse normalize method "{self.norm_type}" not implemented'
            )


class BaseData(object):

    def set_num_processes(self, n_proc):

        if (n_proc is None) or (n_proc <= 0):
            self.n_proc = cpu_count()  # max(1, cpu_count() - 1)
        else:
            self.n_proc = min(n_proc, cpu_count())


class SINDData(BaseData):
    """
    Dataset class for SIND dataset.
    Attributes:
        all_df: dataframe indexed by ID, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, config: dict, n_proc=None):

        n_proc = config["n_proc"] if n_proc is None else n_proc
        self.set_num_processes(n_proc=n_proc)
        self.config = config
        self.feature_names = None   # inferred dynamically from CSV headers
        self.all_df = None
        self.all_IDs = None
        self.feature_df = None
        self.max_seq_len = self.config["data_chunk_len"]

    def load_data(self):
            self.file_paths = self._gather_data_paths(self.config["data_dir"], pattern=self.config["pattern"])
            self.max_seq_len = self.config["data_chunk_len"] if self.config["data_chunk_len"] > 0 else 50
            # eval_subset takes precedence during clustering runs, then data_subset
            max_chunks = self.config.get("eval_subset") or self.config.get("data_subset") or None

            # ── Infer feature names by majority vote across first 20 files ──────
            import pandas as pd
            from collections import Counter
            _probe_paths = self.file_paths[:20]
            _col_counts = Counter()
            _col_sets = {}
            for _p in _probe_paths:
                _h = pd.read_csv(_p, nrows=0)
                _cols = tuple(c for c in _h.columns if c != "Time")
                _col_counts[_cols] += 1
            # Use the most common column set
            _winner_cols = _col_counts.most_common(1)[0][0]
            self.feature_names = list(_winner_cols)
            logger.info(
                f"Feature names inferred (majority vote, {len(_probe_paths)} probed): "
                f"{len(self.feature_names)} features"
            )
            del _probe_paths, _col_counts, _winner_cols

            import gc
            _chunk_list = []
            done = False

            skipped = 0
            for filepath in self.file_paths:
                if done:
                    break
                df = self.load_single(filepath)

                # Skip files that don't have all expected feature columns
                missing = [c for c in self.feature_names if c not in df.columns]
                if missing:
                    skipped += 1
                    del df
                    continue

                for track_id, group in df.groupby("track_id"):
                    track_data = group[self.feature_names].astype(np.float32).values
                    num_frames = len(track_data)
                    for start_idx in range(0, num_frames, self.max_seq_len):
                        end_idx = start_idx + self.max_seq_len
                        chunk = track_data[start_idx:end_idx]
                        if len(chunk) > 1:
                            # Pad short trailing chunks to uniform length
                            if len(chunk) < self.max_seq_len:
                                pad = np.zeros(
                                    (self.max_seq_len - len(chunk), len(self.feature_names)),
                                    dtype=np.float32,
                                )
                                chunk = np.vstack([chunk, pad])
                            _chunk_list.append(chunk)
                        if max_chunks is not None and len(_chunk_list) >= max_chunks:
                            done = True
                            break
                    if done:
                        break

                del df
                gc.collect()

            self.all_chunks = _chunk_list
            self.all_IDs = list(range(len(self.all_chunks)))
            logger.info(
                f"Loaded {len(self.all_chunks)} chunks "
                f"({skipped} files skipped — column mismatch)."
            )

    
    def _gather_data_paths(self, root_dir, pattern):
        # Implementation to gather data paths  based on a given pattern

        data_paths = []  # list of all paths
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                data_paths.append(os.path.join(root, file))

        if len(data_paths) == 0:
            raise Exception(
                "No files found using: {}".format(os.path.join(root_dir, "*"))
            )

        if pattern is None:
            # by default evaluate on
            selected_paths = data_paths
        else:
            selected_paths = list(filter(lambda x: re.search(pattern, x), data_paths))

        input_paths = [
            p for p in selected_paths if os.path.isfile(p) and p.endswith(".csv")
        ]
        if len(input_paths) == 0:
            raise Exception("No .csv files found using pattern: '{}'".format(pattern))

        return input_paths

    @staticmethod
    def load_single(filepath):
        df = SINDData.read_data(filepath)
        df = SINDData.sort_clean_data(df)
        num_nan = df.isna().sum().sum()
        if num_nan > 0:
            df = df.fillna(1000)  # NAN VALUES TO 1000
        return df

    @staticmethod
    def read_data(filepath):
        """Reads a single .csv, which typically contains a set of datasets of various machine sessions."""
        file_name = os.path.basename(filepath).split('.')[0]
        df = pd.read_csv(filepath)
        df["file_id"] = file_name

        return df

    @staticmethod
    def sort_clean_data(df):
        """"""
        if "track_id" not in df.columns:
            df["track_id"] = df.get("VehicleID", df.get("car_id", df["file_id"]))

        # Keep track_id + Time + ALL feature columns present in this file (dynamic)
        meta_cols = {"track_id", "Time", "file_id", "VehicleID", "car_id"}
        feature_cols = [c for c in df.columns if c not in meta_cols]
        keep_cols = ["track_id", "Time"] + feature_cols

        # Factorize non-numeric columns like LaneID into integers
        if 'LaneID' in df.columns:
            df['LaneID'] = pd.factorize(df['LaneID'])[0]

        # sort based on time and id
        df_sorted = df.sort_values(by=["track_id", "Time"])

        # make track id unique among different files
        df_sorted["track_id"] = (
            df_sorted["file_id"].astype(str) + "_" + df_sorted["track_id"].astype(str)
        )

        # keep columns (only those that exist in the df)
        keep_cols = [c for c in keep_cols if c in df_sorted.columns]
        df_final = df_sorted[keep_cols]

        # remove_stationary_trajectories efficiently without massive memory transform
        has_speed = df_final.groupby("track_id")["Speed"].transform("max") > 0
        df_final = df_final[has_speed]

        return df_final

    def _gather_data_paths(self, root_dir, pattern):
        # Implementation to gather data paths  based on a given pattern

        data_paths = []  # list of all paths
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                data_paths.append(os.path.join(root, file))

        if len(data_paths) == 0:
            raise Exception(
                "No files found using: {}".format(os.path.join(root_dir, "*"))
            )

        if pattern is None:
            # by default evaluate on
            selected_paths = data_paths
        else:
            selected_paths = list(filter(lambda x: re.search(pattern, x), data_paths))

        input_paths = [
            p for p in selected_paths if os.path.isfile(p) and p.endswith(".csv")
        ]
        if len(input_paths) == 0:
            raise Exception("No .csv files found using pattern: '{}'".format(pattern))

        return input_paths

    @staticmethod
    def assign_chunk_idx(df, chunk_len):
        """Assigns a chunk index to each row and trajectory."""
        if chunk_len <= 0:
            chunk_len = 50  # Fallback to safe sequence length
            
        # Calculate local chunk indices within each unique trajectory safely
        df["chunk_idx"] = df.groupby("track_id").cumcount() // chunk_len

        # Generate a global chunk ID by enumerating each unique combination of unique_int_id and chunk_idx
        df["data_chunk_len"] = df.groupby(
            ["track_id", "chunk_idx"]
        ).ngroup()  # ngroup assigns unique numbers to each group

        return df

    @staticmethod
    def remove_small_chunks(df, min_size=2):
        """
        Removes chunks from the dataframe that have fewer than min_size points.

        Parameters:
        - df: The dataframe to process.
        - min_size: The minimum number of points a chunk must have to be retained.

        Returns:
        - The filtered dataframe.
        """
        # Group by global_chunk_id and filter
        filtered_df = df.groupby("data_chunk_len").filter(lambda x: len(x) >= min_size)
        return filtered_df

    def reassign_chunk_indices(self, df):
        # Create a unique list of the old chunk indices
        unique_chunks = df["data_chunk_len"].unique()
        # Create a mapping from old to new indices
        chunk_mapping = {
            old_idx: new_idx for new_idx, old_idx in enumerate(unique_chunks)
        }
        # Map the old indices to new indices
        df["data_chunk_len"] = df["data_chunk_len"].map(chunk_mapping)
        return df


data_factory = {"sind": SINDData}
