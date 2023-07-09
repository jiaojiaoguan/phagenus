#coding=utf-8

import os
import subprocess
from Bio import SeqIO

# Contig2Sentence
def get_mcl_protein_cluster(protein_cluster_database):
    dict_database_protein_pc = {}
    for lines in open(protein_cluster_database):
        line = lines.strip().split(",")
        protein = line[0]
        pc_id = int(line[1])
        dict_database_protein_pc[protein] = pc_id
    return dict_database_protein_pc


def convert_sequence_to_protein_sentences(protein_cluster_database,diamond_out_fp,prodigal_out_protein,out_fn):

    dict_database_protein_pc = get_mcl_protein_cluster(protein_cluster_database)
    dict_protein_pc = {}
    dict_protein_pc_bitscore = {}
    dict_protein_max_bitscore = {}
    for lines in open(diamond_out_fp):
        line = lines.strip().split("\t")
        if line[0] not in dict_protein_max_bitscore.keys():
            max_bitscore = float(line[-1])
            dict_protein_max_bitscore[line[0]] = max_bitscore
        else:
            continue

    percent = 0.98
    for lines in open(diamond_out_fp):
        line = lines.strip().split("\t")
        top_bitscore = dict_protein_max_bitscore[line[0]] * percent
        if float(line[-1]) >= top_bitscore:
            if line[0] not in dict_protein_pc.keys():
                dict_protein_pc[line[0]] = []
            pcid = dict_database_protein_pc[line[1]]
            dict_protein_pc[line[0]].append(pcid)
            if (line[0], pcid) not in dict_protein_pc_bitscore.keys():
                dict_protein_pc_bitscore[(line[0], pcid)] = 0
            dict_protein_pc_bitscore[(line[0], pcid)] += float(line[-1])

    dict_protein_final_pc = {}
    for k, v in dict_protein_pc.items():
        bit_score = []
        for pc in v:
            bit_score.append(dict_protein_pc_bitscore[(k, pc)])
        index = bit_score.index(max(bit_score))
        dict_protein_final_pc[k] = v[index]

    dict_contig_protein = {}
    for lines in open(prodigal_out_protein):
        if lines[0] == ">":
            line = lines.strip().split("#")
            protein = line[0][1:].strip()
            c = protein.split("_")
            length = len(c)
            contig = ""
            for i in range(0, length - 1):
                if i == length - 2:
                    contig = contig + c[i]
                else:
                    contig = contig + c[i] + "_"

            if contig not in dict_contig_protein.keys():
                dict_contig_protein[contig] = []
            dict_contig_protein[contig].append(protein)

    write_protein_file1 = f"{out_fn}/test_protein_sentence.csv"
    file1 = open(write_protein_file1, "w")
    file1.write("accession" + "\t" + "feature" + "\n")

    for k, v in dict_contig_protein.items():
        all_protein = v
        protein_sentence=[]
        for p in all_protein:
            try:
                protein_sentence.append(str(dict_protein_final_pc[p]))
            except KeyError:
                continue
        if len(protein_sentence)>0:
            file1.write(k + "\t")
            for ps in protein_sentence[:-1]:
                file1.write(ps+" ")
            file1.write(protein_sentence[-1])
        file1.write("\n")
    file1.close()
    return 1


def preprocessing_data(input_data,midfolder,similarity,length,threads):
    #Check folders
    out_fn = midfolder
    if not os.path.isdir(out_fn):
        os.makedirs(out_fn)

    #Filter short contigs
    rec = []
    for record in SeqIO.parse(input_data, 'fasta'):
        if len(record.seq) > length:
            rec.append(record)
    SeqIO.write(rec, f'{out_fn}/filtered_contigs.fa', 'fasta')

    #Prodigal translation
    prodigal_cmd = f'prodigal -i {out_fn}/filtered_contigs.fa -a {out_fn}/test_protein.fa -f gff -p meta'
    print(f"Running prodigal...")
    _ = subprocess.check_call(prodigal_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    prodigal_out_protein=f"{out_fn}/test_protein.fa"

    #DIAMOND BLASTP
    print("\n\n" + "{:-^80}".format("Diamond BLASTp"))
    if similarity=="high":
        database="data/high_similarity_complete_genome"
        protein_cluster_database="data/protein_cluster_high_similarity.csv"

    elif similarity=="low":
        database="data/low_similarity_complete_genome"
        protein_cluster_database="data/protein_cluster_low_similarity.csv"
    else:
        print("You do not choose database!")
        exit(0)

    # running alignment
    diamond_cmd = f'diamond blastp -q {out_fn}/test_protein.fa -d {database}.dmnd --threads {threads} -o {out_fn}/diamond_results.tab --sensitive'
    print("Running Diamond...")
    _ = subprocess.check_call(diamond_cmd, shell=True)

    diamond_out_fp = f"{out_fn}/diamond_results.tab"

    convert_sequence_to_protein_sentences(protein_cluster_database,diamond_out_fp,prodigal_out_protein,out_fn)





