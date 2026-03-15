unit_pth = "/mnt/sdb/yuran/av_hubert/datasets/lrs2/labels/from_official_pretrained/Layer_12/valid.km"
units = open(unit_pth).readlines()

count_list = []

for unit_line in units:
    unit_line = unit_line.strip().split(' ')
    
    # output "1" for every token
    ones = ["1"] * len(unit_line)

    count_list.append(" ".join(ones) + "\n")

cluster_counts_pth = "/mnt/sdb/yuran/av_hubert/datasets/lrs2/raw/mvlrs_v1/big_data/valid.cluster_counts"

with open(cluster_counts_pth, "w") as f:
    f.write("".join(count_list))
