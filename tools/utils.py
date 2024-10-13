from prettytable import PrettyTable

def print_advanced_table():
    # 创建 PrettyTable 对象
    table = PrettyTable()

    # 设置列名，使用 ' ' 来创建两层标题
    table.field_names = ["T2V Model", "Motion Smoothness Overall", "Motion Smoothness Low", 
                         "Motion Smoothness Mid", "Motion Smoothness High",
                         "Naturalness Overall", "Naturalness Low", "Naturalness Mid", "Naturalness High"]
    
    # 添加数据行
    table.add_row(["FreeNoise [30]", "71.7", "71.7", "95.4", "47.9", "57.1", "54.8", "73.9", "42.5"])
    table.add_row(["GEN-2 [2]", "49.7", "99.5", "49.7", "0.0", "39.1", "81.6", "35.6", "0.0"])
    table.add_row(["OpenSora [22]", "71.5", "95.5", "95.3", "23.7", "49.8", "62.8", "64.2", "22.5"])
    table.add_row(["Pika [4]", "58.0", "99.5", "74.5", "0.0", "39.8", "69.4", "50.1", "0.0"])
    table.add_row(["StreamingT2V [19]", "71.2", "70.9", "95.0", "47.8", "42.2", "44.2", "55.4", "27.0"])
    table.add_row(["VideoCrafter2 [13]", "48.9", "97.8", "48.8", "0.0", "31.4", "70.1", "24.2", "0.0"])
    table.add_row(["H1stX [5]", "47.9", "92.7", "57.3", "0.0", "54.4", "67.5", "54.4", "0.0"])
    
    # 打印表格
    print(table)

# 返回函数以展示其定义和效果
print_advanced_table()