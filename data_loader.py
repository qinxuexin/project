import pandas as pd
import os
from sklearn.model_selection import train_test_split

num_class = 197
pathFileName = '/Users/zhangyiran/Desktop/standfor_car/cars_train/cars_train/'
label = pd.read_excel('/Users/zhangyiran/Desktop/standfor_car/train_label.xlsx')

imgdir = []
labdir = []
paths = os.listdir(pathFileName)
for item in paths[0:]:
    img = pathFileName + item
    imgdir.append(img)

labdir = list(label.values)

x_train, x_vali, y_train, y_vali = train_test_split(imgdir,
                                                    labdir,
                                                    test_size=0.1,
                                                    random_state=0)

if __name__ == "__main__":
    pass
