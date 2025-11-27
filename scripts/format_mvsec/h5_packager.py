import hdf5plugin
import h5py
import numpy as np


class H5Packager:
    def __init__(self, output_path):
        print("Creating file in {}".format(output_path))
        self.output_path = output_path

        self.file = h5py.File(output_path, "w")
        self.event_xs = self.file.create_dataset(
            "events/xs", (0,), dtype=np.dtype(np.int16), maxshape=(None,), chunks=True, 
        )
        self.event_ys = self.file.create_dataset(
            "events/ys", (0,), dtype=np.dtype(np.int16), maxshape=(None,), chunks=True, 
        )
        self.event_ts = self.file.create_dataset(
            "events/ts", (0,), dtype=np.dtype(np.float64), maxshape=(None,), chunks=True, 
        )
        self.event_ps = self.file.create_dataset(
            "events/ps", (0,), dtype=np.dtype(np.bool_), maxshape=(None,), chunks=True,
        )

    def append(self, dataset, data):
        dataset.resize(dataset.shape[0] + len(data), axis=0)
        if len(data) == 0:
            return
        dataset[-len(data) :] = data[:]

    def package_events(self, xs, ys, ts, ps):
        self.append(self.event_xs, xs)
        self.append(self.event_ys, ys)
        self.append(self.event_ts, ts)
        self.append(self.event_ps, ps)

    def package_flow(self, flowmap, timestamp, flow_idx, dt=1):
        flowmap_dset = self.file.create_dataset(
            "flow/dt=" + str(dt) + "/" + "{:09d}".format(flow_idx), data=flowmap, dtype=np.dtype(np.float64),
        )
        flowmap_dset.attrs["size"] = flowmap.shape
        flowmap_dset.attrs["timestamp_from"] = timestamp[0]
        flowmap_dset.attrs["timestamp_to"] = timestamp[1]

    def add_metadata(self, t0, tlast):
        self.file.attrs["t0"] = t0
        self.file.attrs["tk"] = tlast
        self.file.attrs["duration"] = tlast - t0
