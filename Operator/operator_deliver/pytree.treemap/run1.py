from torch.utils._pytree import tree_map

# 定义一个简单的函数
def add_one(x):
    return x + 1

# 嵌套的数据结构
nested_data = {
    'a': [1, 2, 3],
    'b': (4, 5),
    'c': {'d': 6, 'e': 7}
}

# 使用 tree_map 对每个元素应用 add_one 函数
result = tree_map(add_one, nested_data)

print(result)

# if do well there will be correct result
# correct result
# 直接继续引用torch的就可以了
