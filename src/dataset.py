import os
import random
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Dataset-specific organ label mappings
CHAOS_ORGAN_LABELS = {0: "Background", 1: "Liver", 2: "Right Kidney", 3: "Left Kidney", 4: "Spleen"}
SABS_ORGAN_LABELS = {
    0: "Background", 1: "Spleen", 2: "Right Kidney", 3: "Left Kidney", 4: "Gallbladder",
    5: "Esophagus", 6: "Liver", 7: "Stomach", 8: "Aorta", 9: "Inferior Vena Cava",
    10: "Portal Vein and Splenic Vein", 11: "Pancreas", 12: "Right Adrenal Gland", 13: "Left Adrenal Gland"
}

class UnifiedAbdominalDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        dataset_name: str,  # "CHAOS" or "SABS"
        mode: str = "train",
        image_size: int = 256,
        min_mask_pixels: int = 200,
        min_fg_data: int = 200,
        max_cases: Optional[int] = None,
        seed: Optional[int] = None,
        cache_dir: Optional[str] = None,
        preload_volumes: bool = True,
        num_workers: int = 4,
        z_range: int = 3,
        use_fgmask: bool = True,
    ):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.data_dir = Path(data_dir)
        self.dataset_name = dataset_name.upper()
        self.image_size = image_size
        self.min_mask_pixels = min_mask_pixels
        self.min_fg_data = min_fg_data
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.preload_volumes = preload_volumes
        self.num_workers = num_workers
        self.z_range = z_range
        self.use_fgmask = use_fgmask

        # Set organ labels based on dataset
        if self.dataset_name == "CHAOS":
            self.organ_labels = CHAOS_ORGAN_LABELS
            if mode == "train":
                self.target_organs = [1, 2, 3, 4]
                default_max = 3000
            elif mode == "val":
                self.target_organs = [1, 4]
                default_max = 100
            elif mode == "distil":
                self.target_organs = [1, 2, 3, 4]
                default_max = 5000
            else:
                self.target_organs = [1, 2, 3, 4]
                default_max = None
        elif self.dataset_name == "SABS":
            self.organ_labels = SABS_ORGAN_LABELS
            if mode == "train":
                self.target_organs = [1, 2, 3, 6]
                default_max = 3000
            elif mode == "val":
                self.target_organs = [1, 6]
                default_max = 100
            if mode == "distil":
                self.target_organs = list(range(1, 14))
                default_max = 5000
            else:
                self.target_organs = list(range(1, 14))
                default_max = None
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        max_cases = max_cases or default_max

        if self.cache_dir:
            self.cache_dir.mkdir(exist_ok=True, parents=True)

        cache_file = self.cache_dir / f"{self.dataset_name}_{mode}_records.pkl" if self.cache_dir else None

        if cache_file and cache_file.exists():
            logger.info(f"Loading cached records from {cache_file}")
            with open(cache_file, "rb") as f:
                self.case_records = pickle.load(f)
        else:
            logger.info(f"Discovering {self.dataset_name} file sets...")
            case_records = self._find_file_sets()
            # Sort for determinism
            case_records.sort(key=lambda x: x[0])
            logger.info("Validating cases with parallel processing...")
            self.case_records = self._filter_valid_cases_parallel(case_records)
            if cache_file:
                with open(cache_file, "wb") as f:
                    pickle.dump(self.case_records, f)

        if max_cases and len(self.case_records) > max_cases:
            logger.info(f"Sampling {max_cases} cases from {len(self.case_records)} available")
            self.case_records = sorted(random.sample(self.case_records, max_cases), key=lambda x: x[0])

        if len(self.case_records) < 2:
            raise RuntimeError(f"Need at least 2 cases for Support/Query split, found {len(self.case_records)}")

        logger.info("Precomputing valid slice indices...")
        self._precompute_valid_indices()

        if preload_volumes:
            logger.info("Preloading volumes into memory...")
            self._preload_volumes()

    def _find_file_sets(self) -> List[Tuple[int, Dict[str, str]]]:
        from collections import defaultdict
        files_by_number = defaultdict(dict)
        for file in self.data_dir.glob("*.nii"):
            name = file.stem
            parts = name.rsplit('_', 1)
            if len(parts) == 2:
                prefix, number = parts
                try:
                    num = int(number)
                    files_by_number[num][prefix] = file.name
                except ValueError:
                    continue
        complete_sets = []
        for num, file_dict in sorted(files_by_number.items()):
            if 'image' in file_dict and 'label' in file_dict:
                complete_sets.append((num, file_dict))
        return complete_sets

    def _normalize_volume(self, array: np.ndarray) -> np.ndarray:
        """
        Normalize the volume using Mean/Std (Z-score normalization).
        Includes robust clipping to prevent outliers from distorting the std.
        """
        # 1. Clip outliers (0.5% to 99.5%) to stabilize Mean/Std calculation
        p_min = np.percentile(array, 0.5)
        p_max = np.percentile(array, 99.5)
        array = np.clip(array, p_min, p_max)

        # 2. Calculate Mean and Std on the clipped volume
        mean = array.mean()
        std = array.std()

        # 3. Normalize (Z-score)
        if std < 1e-8:
            return np.zeros_like(array, dtype=np.float32)

        return ((array - mean) / std).astype(np.float32)

    def _load_volume(self, filepath: Path, normalize: bool = False) -> np.ndarray:
        """Load volume using SimpleITK and return as numpy array in (x, y, z) format"""
        sitk_img = sitk.ReadImage(str(filepath))
        array = sitk.GetArrayFromImage(sitk_img)

        # SimpleITK returns (z, y, x). We transpose to (x, y, z).

        array = np.transpose(array, (1, 2, 0))

        if normalize:
            array = self._normalize_volume(array)

        return array

    def _is_valid_slice(self, mask_slice: np.ndarray) -> bool:
        return np.count_nonzero(mask_slice) >= self.min_mask_pixels

    def _has_min_fg_data(self, mask_slice: np.ndarray) -> bool:
        return np.count_nonzero(mask_slice) >= self.min_fg_data

    def _validate_case(self, record):
        set_num, file_dict = record
        try:
            label_path = self.data_dir / file_dict['label']
            label = self._load_volume(label_path, normalize=False).astype(np.int16)

            if self.use_fgmask and 'fgmask' in file_dict:
                fgmask_path = self.data_dir / file_dict['fgmask']
                fgmask = self._load_volume(fgmask_path, normalize=False).astype(bool)
                label = label * fgmask

            valid: Dict[int, List[int]] = {}
            for organ in self.target_organs:
                organ_mask = (label == organ)
                slice_counts = np.sum(organ_mask, axis=(0, 1))
                valid_slices = np.where(slice_counts >= self.min_mask_pixels)[0].tolist()
                if len(valid_slices) >= 1:
                    valid[organ] = valid_slices
            return (set_num, file_dict, valid) if valid else None
        except Exception as e:
            logger.warning(f"Error validating case {set_num}: {e}")
            return None

    def _filter_valid_cases_parallel(self, records):
        valids = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
            futures = {ex.submit(self._validate_case, r): r for r in records}
            with tqdm(total=len(records), desc="Validating cases", unit="case") as pbar:
                for f in as_completed(futures):
                    res = f.result()
                    if res: valids.append(res)
                    pbar.update(1)
        return valids

    def _precompute_valid_indices(self):
        self.valid_indices = {i: r[2] for i, r in enumerate(self.case_records)}
        self.organ_to_cases: Dict[int, List[int]] = {}
        for i, organ_map in self.valid_indices.items():
            for organ in organ_map.keys():
                self.organ_to_cases.setdefault(organ, []).append(i)

    def _preload_volumes(self):
        self.volume_cache = {}
        for i, (set_num, file_dict, _) in enumerate(tqdm(self.case_records, desc="Preloading volumes", unit="vol")):
            try:
                # Normalize Image immediately upon loading (3D Normalization)
                img = self._load_volume(self.data_dir / file_dict['image'], normalize=True)

                label = self._load_volume(self.data_dir / file_dict['label'], normalize=False).astype(np.int16)
                fgmask = None
                if self.use_fgmask and 'fgmask' in file_dict:
                    fgmask = self._load_volume(self.data_dir / file_dict['fgmask'], normalize=False).astype(bool)
                self.volume_cache[i] = (img, label, fgmask)
            except Exception as e:
                logger.warning(f"Failed to preload case {i}: {e}")

    def _to_tensor(self, array: np.ndarray, is_mask=False):
        t = torch.from_numpy(array).unsqueeze(0)
        if array.shape == (self.image_size, self.image_size):
            return t
        mode = "nearest" if is_mask else "bilinear"
        t = F.interpolate(
            t.unsqueeze(0), size=(self.image_size, self.image_size),
            mode=mode, align_corners=False if mode == "bilinear" else None
        )
        return t.squeeze(0)

    def __len__(self):
        return len(self.case_records)

    def __getitem__(self, idx):
        max_attempts = 100
        attempt = 0

        # All available patient indices
        all_indices = list(range(len(self.case_records)))

        while attempt < max_attempts:
            attempt += 1

            # 1. Pick Two DIFFERENT Patients Randomly
            # We ignore 'idx' and sample randomly to ensure diversity in pairing
            if len(all_indices) < 2:
                 raise RuntimeError("Dataset needs at least 2 patients.")

            support_case_idx, query_case_idx = random.sample(all_indices, 2)

            # 2. Pick a common organ
            s_organs = set(self.valid_indices[support_case_idx].keys())
            q_organs = set(self.valid_indices[query_case_idx].keys())
            common_organs = list(s_organs.intersection(q_organs))

            if not common_organs:
                continue # Retry if patients don't share any target organ

            organ = random.choice(common_organs)

            # 3. Find Compatible Slices (Z-range constraint)
            # We need a support slice (s) and query slice (q) such that abs(s-q) <= z_range
            s_valid_slices = self.valid_indices[support_case_idx][organ]
            q_valid_slices = self.valid_indices[query_case_idx][organ]

            # Convert to sets for faster lookup
            q_slice_set = set(q_valid_slices)

            # Find all valid (s, q) pairs within range
            valid_pairs = []

            # Optimization: Shuffle support slices to avoid always picking the top one
            # and break early if we find enough candidates
            random.shuffle(s_valid_slices)

            for s_idx in s_valid_slices:
                # Check neighbors: s-3 to s+3
                min_z = s_idx - self.z_range
                max_z = s_idx + self.z_range

                # Intersection of Query slices and the range window
                # This finds q indices that are both valid for the organ AND near s_idx
                candidates = [q for q in range(min_z, max_z + 1) if q in q_slice_set]

                if candidates:
                    q_idx = random.choice(candidates)
                    valid_pairs.append((s_idx, q_idx))
                    break # Found a valid pair, we can stop searching this patient pair

            if not valid_pairs:
                # These two patients don't have this organ at the same height
                continue

            # 4. Select the Pair
            s_idx, q_idx = valid_pairs[0]

            # 5. Load Data
            s_set_num, s_files, _ = self.case_records[support_case_idx]
            q_set_num, q_files, _ = self.case_records[query_case_idx]

            try:
                if hasattr(self, "volume_cache"):
                    s_img, s_label, s_fgmask = self.volume_cache[support_case_idx]
                    q_img, q_label, q_fgmask = self.volume_cache[query_case_idx]
                else:
                    # Fallback disk load (Normalize Image ON LOAD)
                    s_img = self._load_volume(self.data_dir / s_files['image'], normalize=True)
                    s_label = self._load_volume(self.data_dir / s_files['label'], normalize=False).astype(np.int16)
                    q_img = self._load_volume(self.data_dir / q_files['image'], normalize=True)
                    q_label = self._load_volume(self.data_dir / q_files['label'], normalize=False).astype(np.int16)

                    s_fgmask = self._load_volume(self.data_dir / s_files['fgmask'], normalize=False).astype(bool) if self.use_fgmask and 'fgmask' in s_files else None
                    q_fgmask = self._load_volume(self.data_dir / q_files['fgmask'], normalize=False).astype(bool) if self.use_fgmask and 'fgmask' in q_files else None

                # Extract Slices
                s_img_slice = s_img[:, :, s_idx]
                q_img_slice = q_img[:, :, q_idx]

                s_label_slice_raw = s_label[:, :, s_idx]
                q_label_slice_raw = q_label[:, :, q_idx]

                if s_fgmask is not None: s_label_slice_raw *= s_fgmask[:, :, s_idx]
                if q_fgmask is not None: q_label_slice_raw *= q_fgmask[:, :, q_idx]

                s_mask_slice = (s_label_slice_raw == organ).astype(np.uint8)
                q_mask_slice = (q_label_slice_raw == organ).astype(np.uint8)

                # Validation checks (mask size)
                if not (self._is_valid_slice(s_mask_slice) and self._is_valid_slice(q_mask_slice)):
                    continue
                if not (self._has_min_fg_data(s_mask_slice) and self._has_min_fg_data(q_mask_slice)):
                    continue

                # Note: No more per-slice normalization here. Image is already volume-normalized.

                return {
                    "support_image": self._to_tensor(s_img_slice),
                    "support_mask": self._to_tensor(s_mask_slice, True),
                    "query_image": self._to_tensor(q_img_slice),
                    "query_mask": self._to_tensor(q_mask_slice, True),
                    "organ_label": torch.tensor(organ, dtype=torch.long),
                    "support_case_idx": torch.tensor(support_case_idx, dtype=torch.long),
                    "query_case_idx": torch.tensor(query_case_idx, dtype=torch.long),
                    "support_set_num": torch.tensor(s_set_num, dtype=torch.long),
                    "query_set_num": torch.tensor(q_set_num, dtype=torch.long),
                    "dataset": self.dataset_name,
                }

            except Exception as e:
                logger.warning(f"Error processing pair (s={support_case_idx}, q={query_case_idx}): {e}")
                continue

        raise IndexError(f"Failed to generate valid Support/Query pair (Different Patients, Z-range +/-{self.z_range}) "
                         f"after {max_attempts} attempts. Check data alignment or loosen constraints.")


def create_unified_dataloaders(
    chaos_dir: str,
    sabs_dir: str,
    batch_size: int = 1,
    num_workers: int = 2,
    use_both_train: bool = True,
    use_both_val: bool = True,
    **kwargs
):
    """
    Unchanged factory function.
    """
    # Create CHAOS datasets
    logger.info("=" * 60)
    logger.info("Creating CHAOS datasets...")
    logger.info("=" * 60)
    chaos_train_ds = UnifiedAbdominalDataset(chaos_dir, "CHAOS", mode="train", **kwargs)
    chaos_distill_ds = UnifiedAbdominalDataset(chaos_dir, "CHAOS", mode="distil", **kwargs)
    chaos_val_ds = UnifiedAbdominalDataset(chaos_dir, "CHAOS", mode="val", **kwargs)

    # Create SABS datasets
    logger.info("=" * 60)
    logger.info("Creating SABS datasets...")
    logger.info("=" * 60)
    sabs_train_ds = UnifiedAbdominalDataset(sabs_dir, "SABS", mode="train", **kwargs)
    sabs_distill_ds = UnifiedAbdominalDataset(sabs_dir, "SABS", mode="distil", **kwargs)
    sabs_val_ds = UnifiedAbdominalDataset(sabs_dir, "SABS", mode="val", **kwargs)

    # Combine datasets based on flags
    if use_both_train:
        logger.info("\n✓ Combining CHAOS + SABS for training")
        train_ds = ConcatDataset([chaos_train_ds, sabs_train_ds])
        distill_ds = ConcatDataset([chaos_distill_ds, sabs_distill_ds])
    else:
        logger.info("\n✓ Using CHAOS only for training")
        train_ds = chaos_train_ds
        distill_ds = chaos_distill_ds

    if use_both_val:
        logger.info("✓ Combining CHAOS + SABS for validation")
        val_ds = ConcatDataset([chaos_val_ds, sabs_val_ds])
    else:
        logger.info("✓ Using CHAOS only for validation")
        val_ds = chaos_val_ds

    common = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
        "prefetch_factor": 2 if num_workers > 0 else None,
    }

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **common)
    distill_loader = DataLoader(distill_ds, batch_size=batch_size, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False, **common)

    return train_loader, distill_loader, val_loader
