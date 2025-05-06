import pysam

def load_bam(bam_path):
    return pysam.AlignmentFile(bam_path, "rb")

def write_filtered_bam(human_bam_path, output_bam_path, read_names_to_keep):
    input_bam = load_bam(human_bam_path)
    output_bam = pysam.AlignmentFile(output_bam_path, "wb", template=input_bam)
    for read in input_bam.fetch(until_eof=True):
        if read.query_name in read_names_to_keep:
            output_bam.write(read)
    output_bam.close()
    input_bam.close()
