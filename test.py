from ml_nrem import read_edf
from datetime import datetime

#f = "/home/frederic/Projects/ml_nrem/data/S07_W.edf"
#f = "/home/frederic/Downloads/ml_nrem-master/data/S07_W.edf"
f = "/home/frederic/Projects/ml_nrem/data/S07_W_anonymized.edf"
data , fs, chs = read_edf(f)

#import mne
#raw = mne.io.read_raw_edf(f)
#print(raw)
#print(raw.info)
#print(raw.info.ch_names)
#print(raw.annotations)
#print(help(raw.annotations.delete))
#print(raw.annotations.delete(0))
#print(raw.annotations)

from pyedflib import highlevel
data, data_headers, header = highlevel.read_edf(f)
print(header)

anonymize = not True
if anonymize:
    date = datetime(2000, 1, 1, 0, 0, 0)
    highlevel.anonymize_edf(f, 
                            new_file=None, 
                            to_remove=['patientname', 'birth-date', 'startdate'], 
                            new_values=['xxx', '', date], 
                            verify=False, 
                            verbose=False)
