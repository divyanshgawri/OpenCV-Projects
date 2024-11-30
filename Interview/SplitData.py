import os
import random
import shutil
from itertools import islice
outputfolderpath = "Dataset/SplitData"
inputfolderpath = "/home/divyansh/Desktop/Computer_VIsion/OpenCV-Projects/Interview/DataSet/All"
splitRatio = {"train" :0.7,"val":0.2,"test":0.1}
classes = ["Fake","Real"]
try:
    shutil.rmtree(outputfolderpath)
    # print("removed folder")
except OSError as e:
    os.makedirs(outputfolderpath)

# ---------------- Directories to be Create -------------
os.makedirs(f"{outputfolderpath}/train/images",exist_ok=True)
os.makedirs(f"{outputfolderpath}/train/labels",exist_ok=True)
os.makedirs(f"{outputfolderpath}/val/images",exist_ok=True)
os.makedirs(f"{outputfolderpath}/val/labels",exist_ok=True)
os.makedirs(f"{outputfolderpath}/test/images",exist_ok=True)
os.makedirs(f"{outputfolderpath}/test/labels",exist_ok=True)

# ----------Directories to collect ===================
listnames = os.listdir(inputfolderpath)
# print(len(listnames))
uniquenames = []
for names in listnames:
    uniquenames.append(names.split(".")[0])
uniquenames = list(set(uniquenames))
# print((uniquenames))

# print(len(uniquenames))

# =================Shuffle the dataset ===========================
random.shuffle(uniquenames)


# ================Find the number of images for each other ==========
lenData = len(uniquenames)

lenTrain = int(lenData*splitRatio["train"])
lenVal = int(lenData*splitRatio["val"])
lenTest = int(lenData*splitRatio["test"])
# =============== Put the Remaining images in Training ===============================
if lenData != lenTrain+lenVal+lenTest:
    remaining = lenData - (lenTrain+lenVal+lenTest)
    lenTrain+= remaining
    lenVal+= remaining
print(f" Total images : {lenData} \n Split : {lenTrain} {lenVal} {lenTest}")

# =============== Split the List ===============================
lengthToSplit = [lenTrain, lenVal, lenTest]
Input = iter(uniquenames)
output = [list(islice(Input,elem)) for elem in lengthToSplit]
print(f" Total images : {lenData} \n Split : {len(output[0])} {len(output[1])} {len(output[2])}")

# =============== Split the List ===============================
sequence = ['train','val','test']
for i, out in enumerate(output):
    for filename in out:
        image_src = f"{inputfolderpath}/{filename}.jpg"
        label_src = f"{inputfolderpath}/{filename}.txt"
        image_dst = f"{outputfolderpath}/{sequence[i]}/images/{filename}.jpg"
        label_dst = f"{outputfolderpath}/{sequence[i]}/labels/{filename}.txt"

        # Checking if the image file exists before copying
        if os.path.exists(image_src):
            shutil.copy(image_src, image_dst)
        else:
            print(f"Image file not found: {image_src}")

        # Checking if the label file exists before copying
        if os.path.exists(label_src):
            shutil.copy(label_src, label_dst)
        else:
            print(f"Label file not found: {label_src}")
###################### Creating Data.yaml file ##################
outputFolderPath = outputfolderpath
dataYaml = (
    f"path: ../DataSet/SplitData\n"
    f"train: /home/divyansh/Desktop/Computer_VIsion/OpenCV-Projects/Interview/Dataset/SplitData/train/images\n"
    f"val: /home/divyansh/Desktop/Computer_VIsion/OpenCV-Projects/Interview/Dataset/SplitData/val/images\n"
    f"test: /home/divyansh/Desktop/Computer_VIsion/OpenCV-Projects/Interview/Dataset/SplitData/test/images\n"
    f"nc: {len(classes)}\n"
    f"names: {classes}\n"
)

# Write the YAML content to the file
yaml_file_path = f"{outputFolderPath}/data.yaml"
with open(yaml_file_path, 'w') as f:
    f.write(dataYaml)

print(f"data.yaml file created at {yaml_file_path}")