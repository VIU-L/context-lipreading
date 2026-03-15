for split in ['train', 'valid', 'test']:
    unit_pth = f"/mnt/sdb/yuran/av_hubert/datasets/multivsr/labels/from_official_pretrained/Layer_12/{split}.km"
    units = open(unit_pth).readlines()
    count_list = []
    for unit_line in units:
        unit_line = unit_line.strip().split(' ')
        int_unit_line = [int(x) for x in unit_line]
        current_count = 1
        counts = []
        for i in range(1, len(int_unit_line)):
            if int_unit_line[i] == int_unit_line[i - 1]:
                current_count += 1
            else:
                counts.append(current_count)
                current_count = 1
        counts.append(current_count)
        str_counts = [str(x) for x in counts]
        count_list.append(' '.join(str_counts) + '\n')
    output_pth=f"/mnt/sdb/yuran/av_hubert/datasets/multivsr/lrs3_format/data/{split}.cluster_counts"
    with open(output_pth, 'w') as f:
        f.write(''.join(count_list))