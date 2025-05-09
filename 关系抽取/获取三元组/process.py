
"""加载baike_triples.txt中的三元组"""
file = 'matched_triples.txt'
output_file = 'filted_triples.txt'
triples = []
with open(output_file, "w", encoding="utf-8") as f_out:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")  # 假设三元组是以 tab 分隔
            if len(parts) == 3:
                if parts[0] != parts[2]:
                    f_out.write("\t".join(parts) + "\n")



