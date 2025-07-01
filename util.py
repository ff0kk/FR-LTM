import s2sphere
import pandas as pd
import numpy as np
from tqdm import tqdm
import math
import collections
import json
from random import randint, shuffle, choice
from random import random as rand
from functools import partial
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import shapely.geometry as geometry
import s2sphere
import inspect
import torch
from torch import nn
from geopy.distance import geodesic
from typing import Callable, List, Optional, Set, Tuple, Union
import os
import shutil, stat

def compute_dist(latLgt_1, latLgt_2):
    return geodesic(latLgt_1, latLgt_2).m  # 单位：m

def load_grid(grid_file):
    """Loads a grid_file file into a dictionary."""
    with open(grid_file, "r", encoding="UTF-8") as file:
        cellID2gridID = json.load(file)
    return cellID2gridID
    
def save_grid(grid_file, cellID2gridID):
    json.dump(cellID2gridID, open(grid_file, "w"), ensure_ascii=False)

def round_cmit(data):
    data_floor = math.floor(data)
    if data - data_floor >= 0.5:
        data_floor += 1
    return data_floor

def filterByCity(gpd, cityName, isMunicipality=False):
    if isMunicipality is False:
        return gpd[gpd["City"] == cityName]
    else:
        return gpd[gpd["Province"] == cityName]

def withinCity(gpd, lng, lat):
    point = geometry.Point(lng, lat)
    flag_within = False
    for area in gpd.geometry.to_list():
        if point.within(area):
            flag_within = True
            break
        else:
            continue
    return flag_within
    
def getAdminDistIds(gpd, lng, lat):
    point = geometry.Point(lng, lat)
    QXQHDM, DSQHDM, ID = -1, -1, -1
    QXQHDM_list, DSQHDM_list, ID_list = gpd.QXQHDM.to_list(), gpd.DSQHDM.to_list(), gpd.ID.to_list()
    geometry_list = gpd.geometry.to_list()
    for idx in range(len(geometry_list)):
        if point.within(geometry_list[idx]):
            QXQHDM, DSQHDM, ID = int(QXQHDM_list[idx]), int(DSQHDM_list[idx]), int(ID_list[idx])
            break
        else:
            continue
    return (QXQHDM, DSQHDM, ID)

class GoogleS2Util(object):

    # (lat, lng, level=19) -> cell ID
    @classmethod
    def latlng2CellID(cls, lat, lng, level=19):
        ll = s2sphere.LatLng.from_degrees(lat, lng)
        cellId = s2sphere.CellId.from_lat_lng(ll).parent(level).id()  # 指定级数
        return cellId

    # cell ID -> (lat, lng)
    @classmethod
    def cellID2Latlng(cls, cellId):
        ll = s2sphere.CellId(cellId).to_lat_lng()
        point = [ll.lat().degrees, ll.lng().degrees]
        return point

    # cell ID -> (lat, lng) 
    @classmethod
    def cellID2Latlng2(cls, cellId):
        cell = s2sphere.Cell(s2sphere.CellId(cellId))
        ll = s2sphere.LatLng.from_point(cell.get_center())
        point = [ll.lat().degrees, ll.lng().degrees]
        return point

    # Determine the inclusion relationship between cells:      (cellIdA, cellIdB) -> isContain
    @classmethod
    def cellIsContain(cls, cellIdA, cellIdB):
        return s2sphere.CellId(cellIdA).contains(s2sphere.CellId(cellIdB))

    # Get the four positions of the cell:       cell ID -> List<GeoPoint>
    @classmethod
    def cellId2Vertexs(cls, cellId):
        cell = s2sphere.Cell(s2sphere.CellId(cellId))
        points = []
        for i in range(4):
            ll = s2sphere.LatLng.from_point(cell.get_vertex(i))
            points.append([ll.lat().degrees, ll.lng().degrees])
        return points

    # Calculate the distance between two points (unit: meter)： (lat1, lng1, lat2, lng2) -> diatance
    @classmethod
    def getDistance(cls, lat1, lng1, lat2, lng2):
        latLng1 = s2sphere.LatLng.from_degrees(lat1, lng1)
        latLng2 = s2sphere.LatLng.from_degrees(lat2, lng2)
        angle = latLng1.get_distance(latLng2)
        distance = angle.radians * 6371.01 * 1000
        return distance

    @classmethod
    def rad(self, ang):
        return math.pi * ang / 180.0
    @classmethod
    def getDistance2(cls, lat1, lng1, lat2, lng2):
        lat1 = cls.rad(lat1)
        lng1 = cls.rad(lng1)
        lat2 = cls.rad(lat2)
        lng2 = cls.rad(lng2)
        dlng = lng2 - lng1
        dlat = lat2 - lat1
        a = (math.sin(dlat / 2)) ** 2
        b = math.cos(lat1) * math.cos(lat2)
        c = (math.sin(dlng / 2)) ** 2
        distance = 6371.01 * 2 * math.asin(math.sqrt(a + b * c)) * 1000.0
        return distance

    @classmethod
    def getLevel(cls, cellId):
        cellID = s2sphere.CellId(cellId)
        return cellID.level()

    @classmethod
    def getNeighborCells(cls, cellId, level=None):
        if level is None:
            level = cls.getLevel(cellId)
        cellID = s2sphere.CellId(cellId)
        cellANeighbor = [item.id() for item in cellID.get_all_neighbors(level)]
        return cellANeighbor

    @classmethod
    def isNeighbor(cls, cellIdA, cellIdB):
        return cellIdA in cls.getNeighborCells(cellIdB)

    @classmethod
    def getCoverCells(cls, lat1, lng1, lat2, lng2, minLevel=0, maxLevel=30, maxCells=100):
        startS2 = s2sphere.LatLng.from_degrees(min(lat1, lat2), min(lng1, lng2))
        endS2 = s2sphere.LatLng.from_degrees(max(lat1, lat2), max(lng1, lng2))
        rect = s2sphere.LatLngRect(startS2, endS2)  # 矩形区域

        s2RegionCoverer = s2sphere.RegionCoverer()
        s2RegionCoverer.min_level = minLevel
        s2RegionCoverer.max_level = maxLevel
        s2RegionCoverer.max_cells = maxCells

        covering = s2RegionCoverer.get_covering(rect)
        return [item.id() for item in covering]
    
    @classmethod
    def cellId2exact_rea(cls, cellId):
        cell = s2sphere.Cell(s2sphere.CellId(cellId))
        return cell.exact_area()
    
    @classmethod
    def cellId2approx_area(cls, cellId):
        cell = s2sphere.Cell(s2sphere.CellId(cellId))
        return cell.approx_area()
    
    @classmethod
    def cellId2center(cls, cellId):
        return np.mean(cls.cellId2Vertexs(cellId), axis=0)

def cidLocation2cellInfo(cid_location, filtered_gpd, gridLevel=19):
    filtered_cid_location = []
    for cid, lat, lng in tqdm(zip(cid_location.CID.to_list(), cid_location.latitude.to_list(), cid_location.longitude.to_list())):
        if lat > 0.0 and lng > 0.0:
            adminDistIds = getAdminDistIds(filtered_gpd, lng, lat)
            if adminDistIds[0] != -1 and  adminDistIds[1] != -1 and  adminDistIds[2] != -1:
                filtered_cid_location.append([cid, lat, lng, adminDistIds[0], adminDistIds[1], adminDistIds[2]])
            else:
                continue
        else:
            continue
            
    cid_location_pd = pd.DataFrame(filtered_cid_location, columns=["CID", "latitude", "longitude", "QXQHDM", "DSQHDM", "ID"])
    
    cellId_list = []
    for lat, lng in tqdm(zip(cid_location_pd.latitude.to_list(), cid_location_pd.longitude.to_list())):
        cellId_list.append(GoogleS2Util.latlng2CellID(lat, lng, gridLevel))
    
    cid_location_pd["cellId"] = cellId_list
    
    cellId_list.sort()
    cellId2sortedCellId = {cellId: i for i, cellId in enumerate(cellId_list)}
    
    cid_location_pd["sortedCellId"] = cid_location_pd["cellId"].map(cellId2sortedCellId)

    lat_cellCenter_list, lng_cellCenter_list, cellArea_list = [], [], []
    for cellId in tqdm(cid_location_pd.cellId.to_list()):
        lat, lng = GoogleS2Util.cellId2center(cellId)
        cellArea = GoogleS2Util.cellId2exact_rea(cellId)
        lat_cellCenter_list.append(lat)
        lng_cellCenter_list.append(lng)
        cellArea_list.append(cellArea)
    
    cid_location_pd["lat_cellCenter"] = lat_cellCenter_list
    cid_location_pd["lng_cellCenter"] = lng_cellCenter_list
    cid_location_pd["cellArea"] = cellArea_list
    
    return cid_location_pd

def batch_list_to_batch_tensors(batch):
    batch_tensors = []
    for x in zip(*batch):
        if x[0] is None:
            batch_tensors.append(None)
        elif isinstance(x[0], torch.Tensor):
            batch_tensors.append(torch.stack(x))
        else:
            batch_tensors.append(torch.tensor(x))
    return batch_tensors


def find_pruneable_heads_and_indices(
    heads: List[int], n_heads: int, head_size: int, already_pruned_heads: Set[int]
) -> Tuple[Set[int], torch.LongTensor]:
    """
    Finds the heads and their indices taking `already_pruned_heads` into account.

    Args:
        heads (`List[int]`): List of the indices of heads to prune.
        n_heads (`int`): The number of heads in the model.
        head_size (`int`): The size of each head.
        already_pruned_heads (`Set[int]`): A set of already pruned heads.

    Returns:
        `Tuple[Set[int], torch.LongTensor]`: A tuple with the indices of heads to prune taking `already_pruned_heads`
        into account and the indices of rows/columns to keep in the layer weight.
    """
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads  # Convert to set and remove already pruned heads
    for head in heads:
        # Compute how many pruned heads are before the head and move the index accordingly
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index: torch.LongTensor = torch.arange(len(mask))[mask].long()
    return heads, index


def prune_linear_layer(layer: nn.Linear, index: torch.LongTensor, dim: int = 0) -> nn.Linear:
    """
    Prune a linear layer to keep only entries in index.

    Used to remove heads.

    Args:
        layer (`torch.nn.Linear`): The layer to prune.
        index (`torch.LongTensor`): The indices to keep in the layer.
        dim (`int`, *optional*, defaults to 0): The dimension on which to keep the indices.

    Returns:
        `torch.nn.Linear`: The pruned layer as a new layer with `requires_grad=True`.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer


def apply_chunking_to_forward(
    forward_fn: Callable[..., torch.Tensor], chunk_size: int, chunk_dim: int, *input_tensors
) -> torch.Tensor:
    """
    This function chunks the `input_tensors` into smaller input tensor parts of size `chunk_size` over the dimension
    `chunk_dim`. It then applies a layer `forward_fn` to each chunk independently to save memory.

    If the `forward_fn` is independent across the `chunk_dim` this function will yield the same result as directly
    applying `forward_fn` to `input_tensors`.

    Args:
        forward_fn (`Callable[..., torch.Tensor]`):
            The forward function of the model.
        chunk_size (`int`):
            The chunk size of a chunked tensor: `num_chunks = len(input_tensors[0]) / chunk_size`.
        chunk_dim (`int`):
            The dimension over which the `input_tensors` should be chunked.
        input_tensors (`Tuple[torch.Tensor]`):
            The input tensors of `forward_fn` which will be chunked

    Returns:
        `torch.Tensor`: A tensor with the same shape as the `forward_fn` would have given if applied`.


    Examples:

    ```python
    # rename the usual forward() fn to forward_chunk()
    def forward_chunk(self, hidden_states):
        hidden_states = self.decoder(hidden_states)
        return hidden_states


    # implement a chunked forward function
    def forward(self, hidden_states):
        return apply_chunking_to_forward(self.forward_chunk, self.chunk_size_lm_head, self.seq_len_dim, hidden_states)
    ```"""

    assert len(input_tensors) > 0, f"{input_tensors} has to be a tuple/list of tensors"

    # inspect.signature exist since python 3.5 and is a python method -> no problem with backward compatibility
    num_args_in_forward_chunk_fn = len(inspect.signature(forward_fn).parameters)
    if num_args_in_forward_chunk_fn != len(input_tensors):
        raise ValueError(
            f"forward_chunk_fn expects {num_args_in_forward_chunk_fn} arguments, but only {len(input_tensors)} input "
            "tensors are given"
        )

    if chunk_size > 0:
        tensor_shape = input_tensors[0].shape[chunk_dim]
        for input_tensor in input_tensors:
            if input_tensor.shape[chunk_dim] != tensor_shape:
                raise ValueError(
                    f"All input tenors have to be of the same shape: {tensor_shape}, "
                    f"found shape {input_tensor.shape[chunk_dim]}"
                )

        if input_tensors[0].shape[chunk_dim] % chunk_size != 0:
            raise ValueError(
                f"The dimension to be chunked {input_tensors[0].shape[chunk_dim]} has to be a multiple of the chunk "
                f"size {chunk_size}"
            )

        num_chunks = input_tensors[0].shape[chunk_dim] // chunk_size

        # chunk input tensor into tuples
        input_tensors_chunks = tuple(input_tensor.chunk(num_chunks, dim=chunk_dim) for input_tensor in input_tensors)
        # apply forward fn to every tuple
        output_chunks = tuple(forward_fn(*input_tensors_chunk) for input_tensors_chunk in zip(*input_tensors_chunks))
        # concatenate output at same dimension
        return torch.cat(output_chunks, dim=chunk_dim)

    return forward_fn(*input_tensors)


class Pipeline:
    """ Pre-process Pipeline Class : callable """

    def __init__(self):
        super().__init__()
        self.skipgram_prb = None
        self.skipgram_size = None
        self.grids = None
        self.call_count = 0
        self.offline_mode = False
        self.skipgram_size_geo_list = None
        self.span_same_mask = False

    def init_skipgram_size_geo_list(self, p):
        if p > 0:
            g_list = []
            t = p
            for _ in range(self.skipgram_size):
                g_list.append(t)
                t *= (1-p)
            s = sum(g_list)
            self.skipgram_size_geo_list = [x/s for x in g_list]

    def __call__(self, instance):
        raise NotImplementedError

    # pre_whole_grid: tokenize to grids before masking
    # post whole grid (--mask_whole_grid): expand to grids after masking
    def get_masked_pos(self, grids, n_pred, add_skipgram=False, mask_segment=None, protect_range=None):
        pre_grid_split = list(range(0, len(grids)+1))

        span_list = list(zip(pre_grid_split[:-1], pre_grid_split[1:]))

        # candidate positions of masked grids
        cand_pos = []
        special_pos = set()
        if mask_segment:
            for i, sp in enumerate(span_list):
                sp_st, sp_end = sp
                if (sp_end-sp_st == 1) and grids[sp_st].endswith('SEP]'):
                    segment_index = i
                    break
        for i, sp in enumerate(span_list):
            sp_st, sp_end = sp
            if (sp_end-sp_st == 1) and (grids[sp_st].endswith('CLS]') or grids[sp_st].endswith('SEP]')):
                special_pos.add(i)
            else:
                if mask_segment:
                    if ((i < segment_index) and ('a' in mask_segment)) or ((i > segment_index) and ('b' in mask_segment)):
                        cand_pos.append(i)
                else:
                    cand_pos.append(i)
        shuffle(cand_pos)

        masked_pos = set()
        for i_span in cand_pos:
            if len(masked_pos) >= n_pred:
                break
            cand_st, cand_end = span_list[i_span]
            if len(masked_pos)+cand_end-cand_st > n_pred:
                continue
            if any(p in masked_pos for p in range(cand_st, cand_end)):
                continue

            n_span = 1
            rand_skipgram_size = 0
            # ngram
            if self.skipgram_size_geo_list:
                # sampling ngram size from geometric distribution
                rand_skipgram_size = np.random.choice(
                    len(self.skipgram_size_geo_list), 1, p=self.skipgram_size_geo_list)[0] + 1
            else:
                if add_skipgram and (self.skipgram_prb > 0) and (self.skipgram_size >= 2) and (rand() < self.skipgram_prb):
                    rand_skipgram_size = min(
                        randint(2, self.skipgram_size), len(span_list)-i_span)
            for n in range(2, rand_skipgram_size+1):
                tail_st, tail_end = span_list[i_span+n-1]
                if (tail_end-tail_st == 1) and (tail_st in special_pos):
                    break
                if len(masked_pos)+tail_end-cand_st > n_pred:
                    break
                n_span = n
            st_span, end_span = i_span, i_span + n_span

            skip_pos = None

            for sp in range(st_span, end_span):
                for mp in range(span_list[sp][0], span_list[sp][1]):
                    if not(skip_pos and (mp in skip_pos)) and (mp not in special_pos) and not(protect_range and (protect_range[0] <= mp < protect_range[1])):
                        masked_pos.add(mp)

        if len(masked_pos) < n_pred:
            shuffle(cand_pos)
            for pos in cand_pos:
                if len(masked_pos) >= n_pred:
                    break
                if pos not in masked_pos:
                    masked_pos.add(pos)
        masked_pos = list(masked_pos)
        if len(masked_pos) > n_pred:
            # shuffle(masked_pos)
            masked_pos = masked_pos[:n_pred]
        return masked_pos

    def replace_masked_grids(self, grids, masked_pos):
        if self.span_same_mask:
            masked_pos = sorted(list(masked_pos))
        prev_pos, prev_rand = None, None
        for pos in masked_pos:
            if self.span_same_mask and (pos-1 == prev_pos):
                t_rand = prev_rand
            else:
                t_rand = rand()
            if t_rand < 0.8:  # 80%
                grids[pos] = '[MASK]'
            prev_pos, prev_rand = pos, t_rand

def truncate_grids_pair(grids_a, grids_b, max_len):
    if len(grids_a) + len(grids_b) > max_len-3:
        while len(grids_a) + len(grids_b) > max_len-3:
            if len(grids_a) > len(grids_b):
                grids_a = grids_a[:-1]
            else:
                grids_b = grids_b[:-1]
    return grids_a, grids_b

def truncate_grids_signle(grids_a, max_len):
    if len(grids_a) > max_len-2:
        grids_a = grids_a[:max_len-2]
    return grids_a


class BiSTLMDataset4MLM(torch.utils.data.Dataset):
    def __init__(self, file_paths, batch_size, max_len, bi_uni_pipeline=[], cur_epoch=0):
        super().__init__()
        self.max_len = max_len  # maximum length of tokens
        self.bi_uni_pipeline = bi_uni_pipeline
        self.batch_size = batch_size
        
        # read the file into memory
        self.ex_list = []
        file = file_paths[cur_epoch % len(file_paths)]
        file_data = open(file, "r", encoding='utf-8')
        threads = min(8, cpu_count())
        with Pool(threads) as p:
            annotate_ = partial(self.read_data)
            self.ex_list = list(
                tqdm(
                    p.imap(annotate_, file_data.readlines(), chunksize=32),
                    desc="convert traj examples to features",
                )
            )
        print('Load {0} trajectory'.format(len(self.ex_list)))
        
    def read_data(self, line):
        sample = eval(line.strip())
        return (sample["uid"], sample["traj"], sample["position_ids"])
    
    def __len__(self):
        return len(self.ex_list)

    def __getitem__(self, idx):
        instance = self.ex_list[idx]
        new_instance = ()
        for proc in self.bi_uni_pipeline:
            new_instance += proc(instance)
        return new_instance
    
    def __iter__(self):  # iterator to load data
        indices = list(range(len(self.ex_list)))
        shuffle(indices)
        for i in range(0, len(self.ex_list), self.batch_size):
            batch = [self.__getitem__(idx) for idx in indices[i: i+self.batch_size]]
            yield batch_list_to_batch_tensors(batch)


class Preprocess4BiSTLM4MLM(Pipeline):
    """ Pre-processing steps for pretraining transformer """
    def __init__(self, 
                 max_pred, mask_prob, gridder, 
                 max_len=512, skipgram_prb=0, skipgram_size=0):
        super().__init__()
        self.max_len = max_len
        self.max_pred = max_pred  # max grids of prediction
        self.mask_prob = mask_prob  # masking probability
        self.indexer = gridder.grid  # function from grid to grid index
        self.skipgram_prb = skipgram_prb
        self.skipgram_size = skipgram_size
        self.gridder = gridder

    def __call__(self, instance):
        uid, grids, position_ids = instance[:3]
        grids = truncate_grids_signle(grids, self.max_len)
        grids = [str(item) for item in grids]
        grids = ['[CLS]'] + grids + ['[SEP]']
        segment_ids = [0] * len(grids)
        position_ids = truncate_grids_signle(position_ids, self.max_len)
        position_ids = [0] + position_ids + [position_ids[-1]+1]

        # For masked Language Models
        # the number of prediction is sometimes less than max_pred when sequence is short
        effective_length = len(grids) - 2
        n_pred = min(self.max_pred, max(
            1, int(round(effective_length*self.mask_prob))))
        # candidate positions of masked grids
        cand_pos = []
        special_pos = set()
        for i, tk in enumerate(grids):
            if (i < len(grids)+2) and (tk != '[CLS]') and (not tk.startswith('[SEP')):
                cand_pos.append(i)
            else:
                special_pos.add(i)
        shuffle(cand_pos)

        masked_pos = set()
        max_cand_pos = max(cand_pos)
        for pos in cand_pos:
            if len(masked_pos) >= n_pred:
                break
            if pos in masked_pos:
                continue

            if (self.skipgram_prb > 0) and (self.skipgram_size >= 2) and (rand() < self.skipgram_prb):
                # ngram
                cur_skipgram_size = randint(2, self.skipgram_size)
                st_pos, end_pos = pos, pos + cur_skipgram_size
            else:
                # directly mask
                st_pos, end_pos = pos, pos + 1

            for mp in range(st_pos, end_pos):
                if (0 < mp <= max_cand_pos) and (mp not in special_pos):
                    masked_pos.add(mp)
                else:
                    break
        
        # when n_pred < max_pred, we only calculate loss within n_pred
        masked_pos = list(masked_pos)
        if len(masked_pos) > n_pred:
            shuffle(masked_pos)
            masked_pos = masked_pos[:n_pred]

        masked_grids = [grids[pos] for pos in masked_pos]

        # grid Indexing
        masked_ids = self.indexer(masked_grids)
        masked_ids_extended = [-1] * self.max_len

        for pos, idx in zip(masked_pos, masked_ids):
            masked_ids_extended[pos] = idx
            grids[pos] = '[MASK]'
        masked_ids = masked_ids_extended

        # grid Indexing
        input_ids = self.indexer(grids)

        # Zero Padding
        n_pad = self.max_len - len(input_ids)
        input_mask = [1] * len(input_ids)
        input_mask.extend([0]*n_pad)
        input_ids.extend([0]*n_pad)
        segment_ids.extend([0]*n_pad)
        position_ids.extend([0]*n_pad)

        return (input_ids, segment_ids, input_mask, masked_ids, position_ids)
    

class BiSTLMDataset4MLM_DefaultPosIds(torch.utils.data.Dataset):
    def __init__(self, file_paths, batch_size, max_len, bi_uni_pipeline=[], cur_epoch=0):
        super().__init__()
        self.max_len = max_len  # maximum length of tokens
        self.bi_uni_pipeline = bi_uni_pipeline
        self.batch_size = batch_size
        
        # read the file into memory
        self.ex_list = []
        file = file_paths[cur_epoch % len(file_paths)]
        file_data = open(file, "r", encoding='utf-8')
        threads = min(8, cpu_count())
        with Pool(threads) as p:
            annotate_ = partial(self.read_data)
            self.ex_list = list(
                tqdm(
                    p.imap(annotate_, file_data.readlines(), chunksize=32),
                    desc="convert traj examples to features",
                )
            )
        print('Load {0} trajectory'.format(len(self.ex_list)))
        
    def read_data(self, line):
        sample = eval(line.strip())
        return (sample["uid"], sample["traj"])
    
    def __len__(self):
        return len(self.ex_list)

    def __getitem__(self, idx):
        instance = self.ex_list[idx]
        new_instance = ()
        for proc in self.bi_uni_pipeline:
            new_instance += proc(instance)
        return new_instance
    
    def __iter__(self):  # iterator to load data
        indices = list(range(len(self.ex_list)))
        shuffle(indices)
        for i in range(0, len(self.ex_list), self.batch_size):
            batch = [self.__getitem__(idx) for idx in indices[i: i+self.batch_size]]
            yield batch_list_to_batch_tensors(batch)


class LazyBiSTLMDataset4MLM_DefaultPosIds(torch.utils.data.Dataset):
    def __init__(self, file_paths, batch_size, max_len, bi_uni_pipeline=[], cur_epoch=0):
        super().__init__()
        self.max_len = max_len  # maximum length of tokens
        self.bi_uni_pipeline = bi_uni_pipeline
        self.batch_size = batch_size

        self.index = []  
        self.file = file_paths[cur_epoch % len(file_paths)]
        with open(self.file, 'r') as f:
            offset = 0
            for line in tqdm(f):
                self.index.append(offset)
                offset += len(line)
        self.data_size = len(self.index) 
    
    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        offset = self.index[idx]
        with open(self.file, 'r') as f:
            f.seek(offset)
            line = f.readline()
            sample = json.loads(line)
            new_instance = ()
            for proc in self.bi_uni_pipeline:
                new_instance += proc((sample["uid"], sample["traj"]))
            return new_instance
    
    def __iter__(self):  # iterator to load data
        indices = list(range(len(self.data_size)))
        shuffle(indices)
        for i in range(0, len(self.data_size), self.batch_size):
            batch = [self.__getitem__(idx) for idx in indices[i: i+self.batch_size]]
            yield batch_list_to_batch_tensors(batch)


class Preprocess4BiSTLM4MLM_DefaultPosIds(Pipeline):
    """ Pre-processing steps for pretraining transformer """
    def __init__(self, 
                 max_pred, mask_prob, gridder, 
                 max_len=512, skipgram_prb=0, skipgram_size=0,
                 k_minute_timeSlicing=15):
        super().__init__()
        self.max_len = max_len
        self.max_pred = max_pred  # max grids of prediction
        self.mask_prob = mask_prob  # masking probability
        self.indexer = gridder.grid  # function from grid to grid index
        self.skipgram_prb = skipgram_prb
        self.skipgram_size = skipgram_size
        self.gridder = gridder
        self.k_minute_timeSlicing = k_minute_timeSlicing

    def __call__(self, instance):
        uid, grids = instance[:2]
        grids = truncate_grids_signle(grids, self.max_len)
        grids = [str(item) for item in grids]
        grids = ['[CLS]'] + grids + ['[SEP]']
        segment_ids = [0] * len(grids)

        position_ids = list(range(0, (self.max_len - 2)*self.k_minute_timeSlicing, self.k_minute_timeSlicing))
        position_ids = [0] + position_ids + [position_ids[-1]+1]
        effective_length = len(grids) - 2
        n_pred = min(self.max_pred, max(
            1, int(round(effective_length*self.mask_prob))))
        cand_pos = []
        special_pos = set()
        for i, tk in enumerate(grids):
            if (i < len(grids)+2) and (tk != '[CLS]') and (not tk.startswith('[SEP')):
                cand_pos.append(i)
            else:
                special_pos.add(i)
        shuffle(cand_pos)

        masked_pos = set()
        max_cand_pos = max(cand_pos)
        for pos in cand_pos:
            if len(masked_pos) >= n_pred:
                break
            if pos in masked_pos:
                continue

            if (self.skipgram_prb > 0) and (self.skipgram_size >= 2) and (rand() < self.skipgram_prb):
                # ngram
                cur_skipgram_size = randint(2, self.skipgram_size)
                st_pos, end_pos = pos, pos + cur_skipgram_size
            else:
                # directly mask
                st_pos, end_pos = pos, pos + 1

            for mp in range(st_pos, end_pos):
                if (0 < mp <= max_cand_pos) and (mp not in special_pos):
                    masked_pos.add(mp)
                else:
                    break
        
        # when n_pred < max_pred, we only calculate loss within n_pred
        masked_pos = list(masked_pos)
        if len(masked_pos) > n_pred:
            shuffle(masked_pos)
            masked_pos = masked_pos[:n_pred]

        masked_grids = [grids[pos] for pos in masked_pos]

        # grid Indexing
        masked_ids = self.indexer(masked_grids)
        masked_ids_extended = [-1] * self.max_len

        for pos, idx in zip(masked_pos, masked_ids):
            masked_ids_extended[pos] = idx
            grids[pos] = '[MASK]'
        masked_ids = masked_ids_extended

        # grid Indexing
        input_ids = self.indexer(grids)

        # Zero Padding
        n_pad = self.max_len - len(input_ids)
        input_mask = [1] * len(input_ids)
        input_mask.extend([0]*n_pad)
        input_ids.extend([0]*n_pad)
        segment_ids.extend([0]*n_pad)
        position_ids.extend([0]*n_pad)

        return (input_ids, segment_ids, input_mask, masked_ids, position_ids)


def getIds_truncateBySampling(grids_len, max_len):
    ids = list(range(grids_len))
    if grids_len > max_len-2:
        ids = np.random.choice(ids, size=max_len-2, replace=False)
        ids = np.sort(ids)
    return ids

class Preprocess4BiSTLM4TrajClassification_DefaultPosIds(Pipeline):
    def __init__(self, indexer, max_len=512,
                 k_minute=15,
                 is_unidirectional_selfAttention=False):
        super().__init__()
        self.max_len = max_len
        self.indexer = indexer  # function from grid to grid index
        self.k_minute = k_minute
        self.is_unidirectional_selfAttention = is_unidirectional_selfAttention
        if self.is_unidirectional_selfAttention:
            self._tril_matrix = torch.tril(torch.ones(
                (max_len, max_len), dtype=torch.long))

    def __call__(self, instance):
        uid, grids, label, position_ids = instance[:4]
        grids = truncate_grids_signle(grids, self.max_len)

        grids = [str(item) for item in grids]
        grids = ['[CLS]'] + grids + ['[SEP]']
        segment_ids = [0] * len(grids)

        # grid Indexing
        input_ids = self.indexer(grids)

        if position_ids is None:
            position_ids = list(range(0, (len(grids) - 2)*self.k_minute, self.k_minute))
            position_ids = [0] + position_ids + [position_ids[-1]+1]

        # Zero Padding
        n_pad = self.max_len - len(input_ids)
        if not self.is_unidirectional_selfAttention:
            input_mask = [1] * len(input_ids)
            input_mask.extend([0]*n_pad)
            input_ids.extend([0]*n_pad)
        else:
            input_mask = self._tril_matrix
            input_mask[:, :n_pad].fill_(0)
            input_ids = [0]*n_pad + input_ids
        segment_ids.extend([0]*n_pad)

        return (input_ids, position_ids, segment_ids, input_mask, uid, label)
    
class BiSTLMDataset4TrajClassification_DefaultPosIds(torch.utils.data.Dataset):
    def __init__(self, file_paths, batch_size, gridder, max_len, bi_uni_pipeline=[], cur_epoch=0):
        super().__init__()
        self.gridder = gridder  # grid tool object
        self.max_len = max_len  # maximum length of tokens
        self.bi_uni_pipeline = bi_uni_pipeline
        self.batch_size = batch_size

        # read the file into memory
        self.ex_list = []
        file = file_paths[cur_epoch % len(file_paths)]
        file_data = open(file, "r", encoding='utf-8')
        threads = min(8, cpu_count())
        with Pool(threads) as p:
            annotate_ = partial(self.read_data)
            self.ex_list = list(
                tqdm(
                    p.imap(annotate_, file_data.readlines(), chunksize=32),
                    desc="convert traj examples to features",
                )
            )
        print('Load {0} trajectory'.format(len(self.ex_list)))
        
    def read_data(self, line):
        sample = eval(line.strip())
        return (sample["uid"], sample["traj"], #sample["position_ids"],
                int(sample["label"]))
    
    def __len__(self):
        return len(self.ex_list)

    def __getitem__(self, idx):
        instance = self.ex_list[idx]
        new_instance = ()
        for proc in self.bi_uni_pipeline:
            new_instance += proc(instance)
        return new_instance
    
    def __iter__(self):  # iterator to load data
        indices = list(range(len(self.ex_list)))
        shuffle(indices)
        for i in range(0, len(self.ex_list), self.batch_size):
            batch = [self.__getitem__(idx) for idx in indices[i: i+self.batch_size]]
            yield batch_list_to_batch_tensors(batch)


class LazyBiSTLMDataset4TrajClassification_DefaultPosIds(torch.utils.data.Dataset):
    def __init__(self, file_paths, batch_size, gridder, max_len, bi_uni_pipeline=[], cur_epoch=0):
        super().__init__()
        self.gridder = gridder  # grid tool object
        self.max_len = max_len  # maximum length of tokens
        self.bi_uni_pipeline = bi_uni_pipeline
        self.batch_size = batch_size

        self.index = []  
        self.file = file_paths[cur_epoch % len(file_paths)]
        with open(self.file, 'r') as f:
            offset = 0
            for line in tqdm(f):
                self.index.append(offset)
                offset += len(line)
        self.data_size = len(self.index) 
    
    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        offset = self.index[idx]
        with open(self.file, 'r') as f:
            f.seek(offset)
            line = f.readline()
            sample = json.loads(line)
            new_instance = ()
            for proc in self.bi_uni_pipeline:
                if "position_ids" in sample:
                    new_instance += proc((sample["uid"], sample["traj"], int(sample["label"]), sample["position_ids"]))
                else:
                    new_instance += proc((sample["uid"], sample["traj"], int(sample["label"]), None))
            return new_instance
        
    def __iter__(self):  # iterator to load data
        indices = list(range(len(self.data_size)))
        shuffle(indices)
        for i in range(0, len(self.data_size), self.batch_size):
            batch = [self.__getitem__(idx) for idx in indices[i: i+self.batch_size]]
            yield batch_list_to_batch_tensors(batch)


class Preprocess4BiSTLM4TravelModeRecognition(Pipeline):
    def __init__(self, indexer, max_len=512, truncateBySampling=False,
                 is_unidirectional_selfAttention=False):
        super().__init__()
        self.max_len = max_len
        self.indexer = indexer  # function from grid to grid index
        self.truncateBySampling = truncateBySampling
        self.is_unidirectional_selfAttention = is_unidirectional_selfAttention
        if self.is_unidirectional_selfAttention:
            self._tril_matrix = torch.tril(torch.ones(
                (max_len, max_len), dtype=torch.long))

    def __call__(self, instance):
        uid, grids, position_ids, label = instance[:4]
        if self.truncateBySampling:
            grids_truncated, position_ids_truncated, label_truncated = [], [], []
            ids4truncated = getIds_truncateBySampling(len(grids), self.max_len)
            for idx in ids4truncated:
                grids_truncated.append(grids[idx])
                label_truncated.append(label[idx])
                position_ids_truncated.append(position_ids[idx])
            grids, position_ids, label = grids_truncated, position_ids_truncated, label_truncated
        else:
            grids = truncate_grids_signle(grids, self.max_len)
            label = truncate_grids_signle(label, self.max_len)
            position_ids = truncate_grids_signle(position_ids, self.max_len)

        grids = [str(item) for item in grids]
        grids = ['[CLS]'] + grids + ['[SEP]']
        label = [-1] + label + [-1]
        position_ids = [0] + position_ids + [position_ids[-1]+1]
        segment_ids = [0] * len(grids)

        # grid Indexing
        input_ids = self.indexer(grids)

        # Zero Padding
        n_pad = self.max_len - len(input_ids)
        if not self.is_unidirectional_selfAttention:
            input_mask = [1] * len(input_ids)
            input_mask.extend([0]*n_pad)
            input_ids.extend([0]*n_pad)
            position_ids.extend([0]*n_pad)
            label.extend([-1]*n_pad)
        else:
            input_mask = self._tril_matrix
            input_mask[:, :n_pad].fill_(0)
            input_ids = [0]*n_pad + input_ids
            position_ids = [0]*n_pad + position_ids
            label = [-1]*n_pad + label
        segment_ids.extend([0]*n_pad)

        return (input_ids, segment_ids, input_mask, position_ids, uid, label)
    

class BiSTLMDataset4TravelModeRecognition(torch.utils.data.Dataset):
    def __init__(self, file_paths, batch_size, gridder, max_len, bi_uni_pipeline=[], cur_epoch=0,
                 gen_pos_idx=False, k_minute_timeSlicing=-1):
        super().__init__()
        self.gridder = gridder  # grid tool object
        self.max_len = max_len  # maximum length of tokens
        self.bi_uni_pipeline = bi_uni_pipeline
        self.batch_size = batch_size
        self.gen_pos_idx = gen_pos_idx
        self.k_minute_timeSlicing = k_minute_timeSlicing

        # read the file into memory
        self.ex_list = []
        file = file_paths[cur_epoch % len(file_paths)]
        file_data = open(file, "r", encoding='utf-8')
        threads = min(8, cpu_count())
        with Pool(threads) as p:
            annotate_ = partial(self.read_data)
            self.ex_list = list(
                tqdm(
                    p.imap(annotate_, file_data.readlines(), chunksize=32),
                    desc="convert traj examples to features",
                )
            )
        print('Load {0} trajectory'.format(len(self.ex_list)))
        
    def read_data(self, line):
        sample = eval(line.strip())
        if self.gen_pos_idx and self.k_minute_timeSlicing != -1:
            return (sample["uid"], sample["traj"], 
                    list(range(0, (self.max_len-2) * self.k_minute_timeSlicing, self.k_minute_timeSlicing)),
                    sample["travel_mode_label4all"])
        else:
            return (sample["uid"], sample["traj"], sample["position_ids"],
                    sample["travel_mode_label4all"])
    
    def __len__(self):
        return len(self.ex_list)

    def __getitem__(self, idx):
        instance = self.ex_list[idx]
        new_instance = ()
        for proc in self.bi_uni_pipeline:
            new_instance += proc(instance)
        return new_instance
    
    def __iter__(self):  # iterator to load data
        indices = list(range(len(self.ex_list)))
        shuffle(indices)
        for i in range(0, len(self.ex_list), self.batch_size):
            batch = [self.__getitem__(idx) for idx in indices[i: i+self.batch_size]]
            yield batch_list_to_batch_tensors(batch)


class LazyBiSTLMDataset4TravelModeRecognition(torch.utils.data.Dataset):
    def __init__(self, file_paths, batch_size, gridder, max_len, bi_uni_pipeline=[], cur_epoch=0,
                 gen_pos_idx=False, k_minute_timeSlicing=-1):
        super().__init__()
        self.gridder = gridder  # grid tool object
        self.max_len = max_len  # maximum length of tokens
        self.bi_uni_pipeline = bi_uni_pipeline
        self.batch_size = batch_size
        self.gen_pos_idx = gen_pos_idx
        self.k_minute_timeSlicing = k_minute_timeSlicing

        self.index = [] 
        self.file = file_paths[cur_epoch % len(file_paths)]
        with open(self.file, 'r') as f:
            offset = 0
            for line in tqdm(f):
                self.index.append(offset)
                offset += len(line)
        self.data_size = len(self.index)
    
    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        offset = self.index[idx]
        with open(self.file, 'r') as f:
            f.seek(offset)
            line = f.readline()
            sample = json.loads(line)
            new_instance = ()
            for proc in self.bi_uni_pipeline:
                if self.gen_pos_idx and self.k_minute_timeSlicing != -1:
                    new_instance += proc((sample["uid"], sample["traj"], 
                            list(range(0, (self.max_len-2) * self.k_minute_timeSlicing, self.k_minute_timeSlicing)),
                            sample["travel_mode_label4all"]))
                else:
                    new_instance += proc((sample["uid"], sample["traj"], sample["position_ids"],
                            sample["travel_mode_label4all"]))
            return new_instance
    
    def __iter__(self):  # iterator to load data
        indices = list(range(len(self.data_size)))
        shuffle(indices)
        for i in range(0, len(self.data_size), self.batch_size):
            batch = [self.__getitem__(idx) for idx in indices[i: i+self.batch_size]]
            yield batch_list_to_batch_tensors(batch)


class Preprocess4BiSTLM4EmbExport(Pipeline):
    def __init__(self, indexer, max_len=512, 
                 gridder=None, is_for_dynamic_emb=False, associating_tmr_label=False,
                 k_minute=15, associating_userProfiling_label=False):
        super().__init__()
        self.max_len = max_len
        self.indexer = indexer  # function from grid to grid index
        self.gridder = gridder  # grid tool object
        self.is_for_dynamic_emb = is_for_dynamic_emb
        self.associating_tmr_label = associating_tmr_label
        self.k_minute = k_minute
        self.associating_userProfiling_label = associating_userProfiling_label

    def __call__(self, instance):
        if self.is_for_dynamic_emb and self.associating_tmr_label:
            uid, grids, position_ids, travel_mode_label4all = instance[:4]
        elif self.associating_userProfiling_label:
            uid, grids, position_ids, label = instance[:4]
        else:
            uid, grids, position_ids = instance[:3]
        grids = [str(grid) for grid in grids]
        if self.is_for_dynamic_emb:
            gps_list = [self.gridder.cellID2centerGps(int(grid)) if grid[0] != "[" else [-1, -1] for grid in grids]

        grids = truncate_grids_signle(grids, self.max_len)
        grids = ['[CLS]'] + grids + ['[SEP]']
        segment_ids = [0] * len(grids)
        if position_ids is None:
            position_ids = list(range(0, (self.max_len - 2) * self.k_minute, self.k_minute))
        position_ids = [0] + position_ids + [position_ids[-1]+1]
        # grid Indexing
        input_ids = self.indexer(grids)

        # Zero Padding
        n_pad = self.max_len - len(input_ids)
        input_mask = [1] * len(input_ids)
        input_mask.extend([0]*n_pad)
        input_ids.extend([0]*n_pad)
        segment_ids.extend([0]*n_pad)
        position_ids.extend([0]*n_pad)

        if self.is_for_dynamic_emb and self.associating_tmr_label:
            return (input_ids, segment_ids, input_mask, position_ids, uid, gps_list, travel_mode_label4all)
        elif self.is_for_dynamic_emb:
            return (input_ids, segment_ids, input_mask, position_ids, uid, gps_list)
        elif self.associating_userProfiling_label:
            return (input_ids, segment_ids, input_mask, position_ids, uid, label)
        else:
            return (input_ids, segment_ids, input_mask, position_ids, uid)
        

class BiSTLMDataset4EmbExport(torch.utils.data.Dataset):
    def __init__(self, file_paths, batch_size, gridder, max_len, bi_uni_pipeline=[], cur_epoch=0,
                 is_for_dynamic_emb=False, associating_tmr_label=False):
        super().__init__()
        self.gridder = gridder  # grid tool object
        self.max_len = max_len  # maximum length of tokens
        self.bi_uni_pipeline = bi_uni_pipeline
        self.batch_size = batch_size
        self.is_for_dynamic_emb = is_for_dynamic_emb
        self.associating_tmr_label = associating_tmr_label

        # read the file into memory
        self.ex_list = []
        file = file_paths[cur_epoch % len(file_paths)]
        file_data = open(file, "r", encoding='utf-8')
        threads = min(8, cpu_count())
        with Pool(threads) as p:
            annotate_ = partial(self.read_data)
            self.ex_list = list(
                tqdm(
                    p.imap(annotate_, file_data.readlines(), chunksize=32),
                    desc="convert traj examples to features",
                )
            )
        print('Load {0} trajectory'.format(len(self.ex_list)))
        
    def read_data(self, line):
        sample = eval(line.strip())
        if self.is_for_dynamic_emb and self.associating_tmr_label:
            if "position_ids" in sample:
                return (sample["uid"], sample["traj"], sample["position_ids"], 
                        sample["travel_mode_label4all"])
            else:
                return (sample["uid"], sample["traj"], None, 
                        sample["travel_mode_label4all"])
        else:
            if "position_ids" in sample:
                return (sample["uid"], sample["traj"], sample["position_ids"])
            else:
                return (sample["uid"], sample["traj"], None)
    
    def __len__(self):
        return len(self.ex_list)

    def __getitem__(self, idx):
        instance = self.ex_list[idx]
        new_instance = ()
        for proc in self.bi_uni_pipeline:
            new_instance += proc(instance)
        return new_instance
    
    def __iter__(self):  # iterator to load data
        indices = list(range(len(self.ex_list)))
        shuffle(indices)
        for i in range(0, len(self.ex_list), self.batch_size):
            batch = [self.__getitem__(idx) for idx in indices[i: i+self.batch_size]]
            yield batch_list_to_batch_tensors(batch)


class LazyBiSTLMDataset4EmbExport(torch.utils.data.Dataset):
    def __init__(self, file_paths, batch_size, gridder, max_len, bi_uni_pipeline=[], cur_epoch=0,
                 is_for_dynamic_emb=False, associating_tmr_label=False,
                 associating_userProfiling_label=False):
        super().__init__()
        self.gridder = gridder  # grid tool object
        self.max_len = max_len  # maximum length of tokens
        self.bi_uni_pipeline = bi_uni_pipeline
        self.batch_size = batch_size
        self.is_for_dynamic_emb = is_for_dynamic_emb
        self.associating_tmr_label = associating_tmr_label
        self.associating_userProfiling_label = associating_userProfiling_label

        self.index = []
        self.file = file_paths[cur_epoch % len(file_paths)]
        with open(self.file, 'r') as f:
            offset = 0
            for line in tqdm(f):
                self.index.append(offset)
                offset += len(line)
        self.data_size = len(self.index) 
    
    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        offset = self.index[idx]
        with open(self.file, 'r') as f:
            f.seek(offset)
            line = f.readline()
            sample = json.loads(line)
            new_instance = ()
            for proc in self.bi_uni_pipeline:
                if self.is_for_dynamic_emb and self.associating_tmr_label:
                    if "position_ids" in sample:
                        new_instance += proc((sample["uid"], sample["traj"], sample["position_ids"], 
                                                sample["travel_mode_label4all"]))
                    else:
                        new_instance += proc((sample["uid"], sample["traj"], None, 
                                                sample["travel_mode_label4all"]))
                elif self.associating_userProfiling_label:
                    if "position_ids" in sample:
                        new_instance += proc((sample["uid"], sample["traj"], sample["position_ids"],
                                              int(sample["label"])))
                    else:
                        new_instance += proc((sample["uid"], sample["traj"], None, int(sample["label"])))
                else:
                    if "position_ids" in sample:
                        new_instance += proc((sample["uid"], sample["traj"], sample["position_ids"]))
                    else:
                        new_instance += proc((sample["uid"], sample["traj"], None))
            return new_instance

    def __iter__(self):  # iterator to load data
        indices = list(range(len(self.data_size)))
        shuffle(indices)
        for i in range(0, len(self.data_size), self.batch_size):
            batch = [self.__getitem__(idx) for idx in indices[i: i+self.batch_size]]
            yield batch_list_to_batch_tensors(batch)


class Preprocess4Left2Right_DefaultPositionIds(Pipeline):
    def __init__(self, indexer, max_len=512, k_minute_timeSlicing=15,
                 ignore_unk=False):
        super().__init__()
        self.max_len = max_len
        self.indexer = indexer
        self._tril_matrix = torch.tril(torch.ones(
            (max_len, max_len), dtype=torch.long))
        self.k_minute_timeSlicing = k_minute_timeSlicing
        self.ignore_unk = ignore_unk

    def __call__(self, instance):
        uid, grids = instance[:2]
        grids = truncate_grids_signle(grids, self.max_len)
        grids = [str(item) for item in grids]
        grids = ['[CLS]'] + grids + ['[SEP]']
        if self.ignore_unk:
            unk_ids = []
            for idx, grid in enumerate(grids):
                if grid == "[UNK]":
                    unk_ids.append(idx)
        segment_ids = [0] * len(grids)

        position_ids = list(range(0, (len(grids) - 2)*self.k_minute_timeSlicing, self.k_minute_timeSlicing))
        position_ids = [0] + position_ids + [position_ids[-1]+1]

        # grid Indexing
        input_ids = self.indexer(grids)

        label_ids = input_ids[::]
        for idx, grid in enumerate(grids):
            if grid == "[UNK]":
                label_ids[idx] = -1
        label_ids = label_ids[1:-1]

        # Zero Padding
        n_pad = self.max_len - len(input_ids)
        input_ids = n_pad * [0] + input_ids
        label_ids = n_pad * [-1] + label_ids + [-1] * 2
        position_ids = n_pad * [-1] + position_ids
        segment_ids = n_pad * [-1] + segment_ids

        input_mask = self._tril_matrix
        input_mask[:, :n_pad].fill_(0)

        if self.ignore_unk:
            for idx in unk_ids:
                input_mask[:, idx].fill_(0)

        return (input_ids, segment_ids, input_mask, label_ids, position_ids)
    

def batch_list_to_batch_tensors_ds_pp(batch):
    batch_tensors = []
    for x in zip(*batch):
        if x[0] is None:
            batch_tensors.append(None)
        elif isinstance(x[0], torch.Tensor):
            batch_tensors.append(torch.stack(x))
        else:
            batch_tensors.append(torch.tensor(x))
    return (tuple(v for v in batch_tensors), batch_tensors[-2])#batch_tensors[-2]为labels

def remove_if_exists(path):
    if os.path.exists(path):
        if os.path.isfile(path) or os.path.islink(path):
            os.remove(path)
        elif os.path.isdir(path):
            def on_rm_error(func, path, exc_info):
                os.chmod(path, stat.S_IWRITE)
                func(path)
            shutil.rmtree(path, onerror=on_rm_error)