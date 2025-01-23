class VDBProposedDataset(Dataset):
    """
    Datasets for the PPG2BPNet model. The same file is used for training and validation == same patients
    """
    def __init__(self, files_dir, filenames, fs, model_fs, segments_duration, standardize_params):
        """
        fs corresponds to the frequency used to generate the data from VitalDB
        """
        self.files_dir = files_dir
        self.filenames = filenames
        self.fs = fs
        self.model_fs = model_fs
        self.segments_length = segments_duration * self.model_fs
        self.mbp_mean = standardize_params["mbp"]["mean"]
        self.mbp_std = standardize_params["mbp"]["std"]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        dfini = pd.read_parquet(os.path.join(self.files_dir, self.filenames[idx]))
        if self.model_fs != self.fs:
            df = dfini.resample("20ms").mean()[:int(len(dfini) / self.fs * self.model_fs)] # frequency defined in PPG2BPNet article
        else:
            df = dfini
        waveforms = df[["SNUADC/PLETH", "SNUADC/ECG_II"]].values
        # patient level scaling 

        dic = df.dic.values[::self.model_fs]
        hr = df["Solar8000/PLETH_HR"].bfill().values[::self.model_fs]
        hr = savgol_filter(hr, 9, 2)
        dic = savgol_filter(dic, 9, 2)
        waveforms = savgol_filter(waveforms, 9, 2, axis=0)
        numerics = np.stack([dic, hr], axis=1)
        mbps = df["mbp"].values[::125]
        cuffs = df["cuff"].values[::125]

        # We reshape the dataset to get segments of 10 seconds of data
        batches = len(df) // self.segments_length
        waveforms = waveforms.reshape(batches, self.segments_length, 2)
        numerics = numerics.reshape(batches, self.segments_length // self.model_fs, 2)
        max_vals = np.max(waveforms, axis=1, keepdims=True)
        min_vals = np.min(waveforms, axis=1, keepdims=True)
        range_vals = max_vals - min_vals
        # Avoid division by zero by setting the range to 1 where it is zero
        range_vals[range_vals == 0] = 1
        waveforms = (waveforms - min_vals) / range_vals
        # scale waveforms to be centered around 0
        waveforms = (waveforms - waveforms.mean(axis=1, keepdims=True)) / waveforms.std(axis=1, keepdims=True)
        # rescale numerics
        max_vals = np.max(numerics, axis=1, keepdims=True)
        min_vals = np.min(numerics, axis=1, keepdims=True)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1
        numerics = (numerics - min_vals) / range_vals
        mbps = mbps.reshape(batches, self.segments_length // self.model_fs)
        cuffs = cuffs.reshape(batches, self.segments_length // self.model_fs)

        # drop 10s segments with NaN indices
        nan_indices_waveforms = np.isnan(waveforms).any(axis=2).any(axis=1)
        nan_indices_numerics = np.isnan(numerics).any(axis=2).any(axis=1)
        nan_indices_mbp = np.isnan(mbps).any(axis=1)
        nan_indices_cuff = np.isnan(cuffs).any(axis=1)
        high_indices_mbp = np.nanmean(mbps, axis=1) > 140
        low_indices_mbp = np.nanmean(mbps, axis=1)  < 40
        low_variance_mbp = np.nanstd(mbps, axis=1) < 1
        low_variance_dic = np.nanstd(numerics[:, :, 0], axis=1) < 0.2
        dic_grad, ecg_grad, hr_grad = np.gradient(dic), np.gradient(waveforms[:, :, 1]), np.gradient(numerics[:, :, 1])
        low_variance_grad_dic = len(np.where(dic_grad == 0)[0]) > 10
        low_variance_grad_ecg = len(np.where(ecg_grad == 0)[0]) > 100
        low_variance_grad_hr = len(np.where(hr_grad == 0)[0]) > 10

        all_bad_indices = nan_indices_waveforms | nan_indices_numerics | nan_indices_mbp | nan_indices_cuff | high_indices_mbp | low_indices_mbp | low_variance_mbp | low_variance_dic | low_variance_grad_dic | low_variance_grad_ecg | low_variance_grad_hr
        waveforms = waveforms[~all_bad_indices, :, :]
        numerics = numerics[~all_bad_indices, :, :]
        mbps = mbps[~all_bad_indices, :]
        cuffs = cuffs[~all_bad_indices, :]

        mbps = (mbps - self.mbp_mean) / self.mbp_std
        cuffs = (cuffs - self.mbp_mean) / self.mbp_std
        
        return waveforms, numerics, mbps, cuffs
