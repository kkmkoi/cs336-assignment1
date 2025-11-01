import numpy as np

# # 修复训练数据
train_path = "/home/huanghy258/zxy23336333/llm/assignment1-basics/tokenized_data/openwebtext_train.npy"
train_data = np.memmap(train_path, dtype=np.uint16, mode='r+')
# train_out_of_range = np.sum((train_data < 0) | (train_data >= 10000))
# print("Train data out of range count (before):", train_out_of_range)

# # 修复超出范围的值
train_data[:] = train_data[:] % 10000

# # 再次检查
# train_out_of_range = np.sum((train_data < 0) | (train_data >= 10000))
# print("Train data out of range count (after):", train_out_of_range)

# # 修复验证数据
val_path = "/home/huanghy258/zxy23336333/llm/assignment1-basics/tokenized_data/openwebtext_valid.npy"
val_data = np.memmap(val_path, dtype=np.uint16, mode='r+')
# val_out_of_range = np.sum((val_data < 0) | (val_data >= 10000))
# print("Validation data out of range count (before):", val_out_of_range)

# # 修复超出范围的值
val_data[:] = val_data[:] % 10000

# # 再次检查
# val_out_of_range = np.sum((val_data < 0) | (val_data >= 10000))
# print("Validation data out of range count (after):", val_out_of_range)

# 检查训练数据
train_data = np.memmap(train_path, dtype=np.uint16, mode='r')
# print("Train data shape:", train_data.shape)
# print("Train data min:", train_data.min())
# print("Train data max:", train_data.max())

# 检查验证数据
val_data = np.memmap(val_path, dtype=np.uint16, mode='r')
# print("Validation data shape:", val_data.shape)
# print("Validation data min:", val_data.min())
# print("Validation data max:", val_data.max())

# 检查训练数据
train_out_of_range = np.sum((train_data < 0) | (train_data >= 32000))
print("Train data out of range count:", train_out_of_range)

# 检查验证数据
val_out_of_range = np.sum((val_data < 0) | (val_data >= 32000))
print("Validation data out of range count:", val_out_of_range)