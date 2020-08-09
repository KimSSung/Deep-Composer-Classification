import py_midicsv
import os
import pandas as pd
import numpy as np

workDir = os.path.abspath("/data/MAESTRO/maestro-v2.0.0")
csv_output_dir = "/data/MAESTRO/midi_csv/"
csv_string = ""
midifile_path_list = []

for dirpath, dirnames, filenames in os.walk(workDir):

    print(dirpath)

    for filename in filenames:
        if filename.endswith(".midi"):
            midifile_path_list.append(str(dirpath) + "/" + str(filename))
            # print("\t", filename)

for index, midifile in enumerate(midifile_path_list):

    csv_string = py_midicsv.midi_to_csv(midifile)

    tmp_list = []

    for i in range(0, len(csv_string)):
        temp = np.array(csv_string[i].replace("\n", "").replace(" ", "").split(","))
        tmp_list.append(temp)
    data = pd.DataFrame(tmp_list)
    data.to_csv(
        csv_output_dir + str(midifile).split("/")[-1] + ".csv", header=True, index=True,
    )
    print(str(midifile).split("/")[-1] + ".csv Saved!")
