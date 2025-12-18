from dataset import PlantCLEFDataset

ds = PlantCLEFDataset("/fs/scratch/PAS2099/plantclef/splits/train.csv")

print("Dataset length:", len(ds))

img, label = ds[0]
print("Sample type:", type(img), type(label))
print("Label:", label)
