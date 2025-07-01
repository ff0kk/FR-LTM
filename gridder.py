from tqdm import tqdm
import collections
import os
import json
import numpy as np

from util import (
        load_grid, save_grid, compute_dist
    )
from util import GoogleS2Util

class Gridder:
    """
    Support gps -> cellID (google s2)
    cellID is equivalent to grid
    Support cellID -> gridID
    """
    def __init__(self,
                 gridding_level=15,
                 cid2cellId_file=None,
                 grid_file=None,
                 lats=None,
                 lngs=None,
                 unk_grid="[UNK]",
                 sep_grid="[SEP]",
                 pad_grid="[PAD]",
                 cls_grid="[CLS]",
                 mask_grid="[MASK]",
                 special_grid_num = 5,
                ):
        self.gridding_level = gridding_level
        self.unk_grid = unk_grid
        self.sep_grid = sep_grid
        self.pad_grid = pad_grid
        self.cls_grid = cls_grid
        self.mask_grid = mask_grid
        self.special_grid_num = special_grid_num
        
        self.cellID2gridID = None
        if grid_file is None or os.path.exists(grid_file) is False:
            if (lats is not None and lngs is not None) or \
                (cid2cellId_file is not None and os.path.exists(cid2cellId_file)):
                self.build_grid_file(lats, lngs, grid_file, cid2cellId_file)
            else:
                raise ValueError(
                    "The grid file path was not passed in & the latitude and longitude of all base stations in the selected city were not passed in"
                )
        else:
            if (lats is not None and lngs is not None) or \
                (cid2cellId_file is not None and os.path.exists(cid2cellId_file)):
                self.build_grid_file(lats, lngs, grid_file, cid2cellId_file)
            else:
                self.cellID2gridID = load_grid(grid_file)
            
        self.gridID2cellID = {gridID: cellID for cellID, gridID in self.cellID2gridID.items()}

        self.compute_lat_lgt_range()

    def compute_lat_lgt_range(self):
        center_gps_list = []
        for cellID in self.cellID2gridID.keys():
            if cellID[0] != "[":
                center_gps_list.append(self.cellID2centerGps(int(cellID)))
            else:
                continue
        center_gps_list = np.array(center_gps_list)
        self.lat_max = max(center_gps_list[:, 0])
        self.lat_min = min(center_gps_list[:, 0])
        self.lgt_max = max(center_gps_list[:, 1])
        self.lgt_min = min(center_gps_list[:, 1])

    def normalize_latLgt(self, latLgt):
        lat, lgt = latLgt[0], latLgt[1]
        return [(lat - ((self.lat_max + self.lat_min) / 2)) / ((self.lat_max - self.lat_min) / 2),
                (lgt - ((self.lgt_max + self.lgt_min) / 2)) / ((self.lgt_max - self.lgt_min) / 2)]
            
    def convert_grid_to_normGps(self, cellIDs):
        normGps_list = []
        for cellID in cellIDs:
            if str(cellID)[0] != "[":
                normGps_list.append(self.normalize_latLgt(self.cellID2centerGps(int(cellID))))
            else:
                normGps_list.append([0, 0])
        
        return normGps_list
    
    def unnormalize_latLgt(self, latLgt):
        lat, lgt = latLgt[0], latLgt[1]
        return [lat * ((self.lat_max - self.lat_min) / 2) + ((self.lat_max + self.lat_min) / 2),
                lgt * ((self.lgt_max - self.lgt_min) / 2) + ((self.lgt_max + self.lgt_min) / 2)]
    
    def convert_normGps_to_gps(self, normGps_list):
        gps_list = []
        for nromGps in normGps_list:
            gps_list.append(self.unnormalize_latLgt(nromGps))
        return gps_list
    
    def compute_dist(self, latLgt_1, latLgt_2):
        return compute_dist(latLgt_1, latLgt_2)
    
    def compute_avg_dist(self, latLgt_list_1, latLgt_list_2):
        dist_list = []
        for latLgt_1, latLgt_2 in zip(latLgt_list_1, latLgt_list_2):
            dist_list.append(self.compute_dist(latLgt_1, latLgt_2))
        return sum(dist_list) / len(dist_list)

    def get_grid_num(self):
        return len(self.cellID2gridID)
    
    def gps2cellID(self, lat, lng):
        return GoogleS2Util.latlng2CellID(lat, lng, self.gridding_level)
    
    def build_grid_file(self, lats, lngs, grid_file, cid2cellId_file):
        self.cellID2gridID = dict()
        if lats is not None and lngs is not None:
            cellIDs = []
            for lat, lng in tqdm(zip(lats, lngs)):
                cellIDs.append(str(self.gps2cellID(lat, lng)))
        else:
            with open(cid2cellId_file, "r", encoding="UTF-8") as file:
                cid2cellId = json.load(file)
            cellIDs = list(cid2cellId.values())
        cellIDs = list(set(cellIDs))
        cellIDs = [self.pad_grid, self.sep_grid, self.unk_grid, self.cls_grid, self.mask_grid] + cellIDs
        for idx, cellID in enumerate(cellIDs):
            self.cellID2gridID[cellID] = idx
        save_grid(grid_file, self.cellID2gridID)
        
    def grid(self, cellIDs):
        return [self.cellID2gridID.get(cellID, self.cellID2gridID[self.unk_grid]) for cellID in cellIDs]
    
    def decode_grid(self, gridIDs):
        return [self.gridID2cellID.get(gridID) for gridID in gridIDs]

    def gps2gridID(self, lat, lng):
        return self.cellID2gridID.get(str(self.gps2cellID(lat, lng)), self.cellID2gridID[self.unk_grid])
        
    def cellID2centerGps(self, cellID):
        return GoogleS2Util.cellId2center(cellID)
    
    def gridID2centerGps(self, gridID):
        if gridID in self.gridID2cellID:
            return self.cellID2centerGps(self.gridID2cellID[gridID])
        else:
            return None
        
    def build_inputs_with_special_grids(self,
                                        grid_ids_0, grid_ids_1=None):
        """
        - single sequence: `[CLS] X [SEP]`
        - pair of sequences: `[CLS] A [SEP] B [SEP]`
        """
        if grid_ids_1 is None:
            return [self.cellID2gridID[self.cls_grid]] + grid_ids_0 + [self.cellID2gridID[self.sep_grid]]
        cls = [self.cellID2gridID[self.cls_grid]]
        sep = [self.cellID2gridID[self.sep_grid]]
        return cls + grid_ids_0 + sep + grid_ids_1 + sep
    
    def get_special_grids_mask(self,
                               grid_ids_0, grid_ids_1=None):
        if grid_ids_0 is not None:
            return [1] + ([0] * len(grid_ids_0)) + [1] + ([0] * len(grid_ids_1)) + [1]
        return [1] + ([0] * len(grid_ids_0)) + [1]
    
    def create_grid_type_ids_from_sequences(self,
                                            grid_ids_0, grid_ids_1=None):
        """
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        """
        cls = [self.cellID2gridID[self.cls_grid]]
        sep = [self.cellID2gridID[self.sep_grid]]
        if grid_ids_1 is None:
            return len(cls + grid_ids_0 + sep) * [0]
        return len(cls + grid_ids_0 + sep) * [0] + len(grid_ids_1 + sep) * [1]