import pandas as pd
from utils.data_processing import encode_categories
from utils.data_processing import train_test_split


data = pd.read_csv("data/stroke.csv", delimiter=",")
processed_data = encode_categories(data)

df_train, df_test = train_test_split(data)
print(df_test)


# for i in range(10000):
#     train, test = train_test_split(data, "stroke", stratify=False, random_seed=i)
#     train_prop, test_prop = test_cases(train, test)
#     train_list.append(train_prop)
#     test_list.append(test_prop)

# print(f"mean train prop: {np.mean(train_list)}")
# print(f"mean test prop: {np.mean(test_list)}")
# print(f"std train prop: {np.std(train_list)}")
# print(f"std test prop: {np.std(test_list)}")
