from __future__ import annotations

from dataclasses import dataclass, asdict, field
import json
from typing import Optional, Tuple, Dict, Any
from enum import Enum
from pathlib import Path
import numpy as np

class InsertionTask(Enum):
    big_rod = "big_rod"
    small_rod = "small_rod"
    rect_rod = "rect_rod"
    usb = "usb"
    ethernet = "ethernet"
    bnc = "bnc"

class ToleranceLevel(Enum):
    tight = 0.1
    medium = 0.2
    loose = 0.3

@dataclass(slots=True)
class RecordConfig:
    script_dir = Path(__file__).parent  # /src/robotics
    project_root = script_dir.parent.parent  # go up to project root
    folder_name: str = ""

    sequence_length: float = None # seconds
    num_insertions: int = None
    sample_rate: float = None #Hz
    deltatime: float = None #seconds
    min_max_deviation_mm: float = None #millimeters
    angular_error_deg: float = None #degrees
    min_max_deviation: float = None #meters
    angular_error: float = None #radians

    insertionTask: InsertionTask = None
    tolerance: ToleranceLevel = None

    holderPose_Up: np.ndarray = None
    holderPose_Down: np.ndarray = None
    insertionPose: np.ndarray = None

    def setSampleRateHz(self, sample_rate: float):
        self.sample_rate = sample_rate
        self.deltatime = 1.0 / sample_rate

    def setSavePath(self, path: str):
        self.folder_name = str(self.project_root / path)
        Path(self.folder_name).mkdir(parents=True, exist_ok=True)

    def setDeviationMM(self, deviation_mm: float):
        self.min_max_deviation_mm = deviation_mm
        self.min_max_deviation = deviation_mm / 1000.0
    
    def setAngularErrorDeg(self, angular_error_deg: float):
        self.angular_error_deg = angular_error_deg
        self.angular_error = np.deg2rad(angular_error_deg)

    def to_dict(self) -> Dict[str, Any]:
        """Convenience for JSON serialization (still basically data-only)."""
        return asdict(self)
    
    def print_config(self):
        print("RecordConfig:")
        for field_name, value in self.to_dict().items():
            print(f"  {field_name}: {value}")

    def getSampleSessionStart(self) -> Tuple[int, int]:
        if not Path(self.folder_name).exists():
            return 0, 0

        max_sample = 0
        session_idx = 0

        for file in Path(self.folder_name).glob("*.csv"):
            parts = file.stem.split("_")
            # print(f"Checking file: {file}, parts: {parts}")
            if len(parts) >= 2:
                try:
                    session = int(parts[3])
                    sample = int(parts[1])
                    if sample > max_sample:
                        max_sample = sample
                        session_idx = session
                except ValueError:
                    continue

        return max_sample, session_idx
    
    def getSFCount(self) -> Tuple[int, int]:
        success = 0
        failure = 0
        for file in Path(self.folder_name).glob("*.json"):
            with open(file, 'r') as f:
                data = json.load(f)
                # print(f"File: {file}, successful_insertion: {data.get('successful_insertion', False)}")
                if data.get("successful_insertion", False):
                    success += 1
                else:
                    failure += 1
        return success, failure