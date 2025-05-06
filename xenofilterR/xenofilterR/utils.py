def count_mismatches(read):
    return read.get_tag("NM") if read and read.has_tag("NM") else 0

def read_pairs(bam):
    read_dict = {}
    for read in bam.fetch(until_eof=True):
        if not read.is_unmapped and read.is_paired:
            if read.query_name not in read_dict:
                read_dict[read.query_name] = [None, None]
            if read.is_read1:
                read_dict[read.query_name][0] = read
            elif read.is_read2:
                read_dict[read.query_name][1] = read
    return read_dict

