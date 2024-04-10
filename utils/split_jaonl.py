import os
import json
import sys

def split_large_jsonl(input_file, output_prefix, total_parts=10):
    # chunk_size_byte = os.path.getsize(input_file) // (total_parts)  # 计算每部分的大致大小（MB）
    # print("chunk_size_byte: ", chunk_size_byte)
    current_part_size = 0
    current_part_num = 0
    current_output_file = None

    def open_next_output_file():
        nonlocal current_part_num, current_output_file
        close_current_output_file()
        current_part_num += 1
        current_output_file = open(f"{output_prefix}_{current_part_num}.jsonl", "w", encoding='utf-8')

    def close_current_output_file():
        if current_output_file:
            current_output_file.close()

    with open(input_file, "r", encoding='utf-8') as large_file:
        large_file_descriptor = large_file.fileno()
        # 使用os.fstat获取文件信息，其中st_size就是文件大小（字节数）
        large_stats = os.fstat(large_file_descriptor)
        large_file_size = large_stats.st_size
        chunk_size_byte = large_file_size // (total_parts) * 8
        print("large_file_size: ", large_file_size)
        print("chunk_size_byte: ", chunk_size_byte)
        for line in large_file:
            # 检查是否需要开启新的输出文件
            if current_output_file is None or current_part_size >= chunk_size_byte:
                open_next_output_file()
                print("current_part_size: ", current_part_size)
                current_part_size = 0

            # 写入当前行并更新当前部分大小
            current_output_file.write(line)
            file_descriptor = current_output_file.fileno()
            # 使用os.fstat获取文件信息，其中st_size就是文件大小（字节数）
            file_stats = os.fstat(file_descriptor)
            file_size = file_stats.st_size
            current_part_size += file_size / 30
            # current_part_size += sys.getsizeof(line.encode())

    # 关闭最后一个输出文件
    close_current_output_file()

# 使用函数
split_large_jsonl('webnovel-chinese/example.jsonl', 'webnovel-chinese/test/output_part')

# 注意：这个脚本假设JSONL文件的行长度相对均匀，以达到近似均分的效果