# this script loads a id flag file, and then downloads from the server the interesting files from it based on the flags
#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob, os

path = "C:/Users/Calder/Outputs/20220821CRfull_20250414T173836/5Hz_30Hz/id_flag.csv"

dir = 'C:\\Users\\Calder\\Outputs\\E7_cleaning\\NORSAR01v2_cleaning_outer\\ID_flags'

paths = glob.glob(os.path.join(dir, '*.csv'))

highlight_interval = pd.Timedelta(minutes=20)
highlight_duration = pd.Timedelta(minutes=4)


for f in paths:
    flags = pd.read_csv(f, sep = ",")
    prefix = flags['file_name'][0].split('_')[0]
    fmt = prefix + '_' + '%Y%m%dT%H%M%S' + 'Z.npy'
    datetimes = pd.to_datetime(flags['file_name'], format = fmt)
    df_filtered = flags.iloc[:,1:-2]
    starttime = datetimes.min()
    endtime = datetimes.max()
    highlight_starts = pd.date_range(start = starttime, end = endtime, freq=highlight_interval)
    event_data = []
    for col in df_filtered.columns:
        times = datetimes[df_filtered[col] !=' '].tolist()
        event_data.append(times)

    # Create the event plot
    plt.figure(figsize=(10, 5))
    plt.eventplot(event_data, orientation='horizontal', linelengths=0.8)
    plt.yticks(range(len(df_filtered.columns)), df_filtered.columns)
    plt.xlabel('Timestamp')
    plt.title('FlagID through time')
    plt.grid(True)
    plt.tight_layout()
    
    for start in highlight_starts:
        end = start+highlight_duration
        if end <= endtime:
            plt.axvspan(start,end,color = 'red', alpha = 0.2)

    name = os.path.split(f)[1].split(sep = '.')[0]
    plotpath = os.path.join(dir,name + '.png')
    plt.title(name)
    #plt.savefig(plotpath)


# def viewflags(path):
#     flags = pd.read_csv(path, sep = ",")

# %%
