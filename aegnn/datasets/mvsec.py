import glob
import numpy as np
import os
import torch

from torch_geometric.data import Data
from torch_geometric.nn.pool import radius_graph
from typing import Callable, List, Optional, Union

from .utils.normalization import normalize_time
from .ncaltech101 import NCaltech101

import pickle


class MVSEC(NCaltech101):

    def __init__(self, batch_size: int = 64, shuffle: bool = True, num_workers: int = 8, pin_memory: bool = False,
                 transform: Optional[Callable[[Data], Data]] = None):
        super(MVSEC, self).__init__(batch_size, shuffle, num_workers, pin_memory=pin_memory, transform=transform)
        self.dims = (120, 100)  # overwrite image shape
        pre_processing_params = {"r": 3.0, "d_max": 32, "n_samples": 10000, "sampling": True}
        self.save_hyperparameters({"preprocessing": pre_processing_params})

    def read_annotations(self, raw_file: str) -> Optional[np.ndarray]:
        return None

    @staticmethod
    def _read_label(raw_file: str) -> List[str]:
        label=["car"]*len(pickle.load(raw_file)['bboxes'])
        return label

    @staticmethod
    def load(raw_file: str) -> Data:
        data=pickle.load(raw_file)
        events=data['events']
        bboxes=data['bboxes']
        x1,y1,x2,y2=bboxes[:,0],bboxes[:,1],bboxes[:,2],bboxes[:,3]
        x_,y_=(x1+x2)/2,(y1+y2)/2
        w,h=abs(x2-x1),abs(y2-y1)
        label=np.array([0]*len(bboxes))
        x,pos=events[:, -1:],events[:,:3]
        #pos[:,2]=normalize_time(pos[:,2])
        bboxes=np.array([x_,y_,w,h,label]).T
        return Data(x=x, pos=pos,bbox=bboxes)

    def pre_transform(self, data: Data) -> Data:
        params = self.hparams.preprocessing

        # Re-weight temporal vs. spatial dimensions to account for different resolutions.
        data.pos[:, 2] = normalize_time(data.pos[:, 2])

        # Coarsen graph by uniformly sampling n points from the event point cloud.
        data = self.sub_sampling(data, n_samples=params["n_samples"], sub_sample=params["sampling"])

        # Radius graph generation.
        data.edge_index = radius_graph(data.pos, r=params["r"], max_num_neighbors=params["d_max"])
        return data

    #########################################################################################################
    # Files #################################################################################################
    #########################################################################################################
    def raw_files(self, mode: str) -> List[str]:
        return glob.glob(os.path.join(self.root, mode, "*"))

    def processed_files(self, mode: str) -> List[str]:
        # random select the dataset in os.path.join(self.root)
        processed_dir = os.path.join(self.root)
        files=os.listdir(processed_dir)
        if mode=='training':
            files=files[:int(len(files)*0.8)]
        else:
            files=files[int(len(files)*0.8):]
        return files
    
    def _load_processed_file(self, f_path: str) -> Data:
        return self.pre_transform(torch.load(f_path))

    @property
    def classes(self) -> List[str]:
        return ["car", "background"]
