"""
This script loads SINMOD and then save the surface values to a csv file for easy loading next time.
"""

folderpath = os.getcwd() + "/../../OneDrive - NTNU/Mascot_PhD/Data/Nidelva/SINMOD_Data/samples/"
filepath = folderpath + "samples_2020.05.04.nc"
files = os.listdir(folderpath)
files.sort()
for file in files:
    if file.endswith(".nc"):
        print(file)
        # filepath = folderpath + file
sinmod = SINMOD(filepath)
data_sinmod = sinmod.get_data()
ind_surface = np.where(data_sinmod[:, 2] == 0.5)[0]
df = np.stack((data_sinmod[ind_surface, 0], data_sinmod[ind_surface, 1], data_sinmod[ind_surface, -1]), axis=1)
ddf = pd.DataFrame(df, columns=["x", "y", "salinity"])
ddf.to_csv("csv/sinmod_" + file[8:18] + ".csv", index=False)

