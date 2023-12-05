import torch
import os
import numpy as np
from torch import nn
import shutil

def check_accuracy_dropout(loader, model,midfolder,var_cutoff,output):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dir_path=midfolder+"tmp/"

    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
        print(f"Deleted directory: {dir_path}")
        os.mkdir(dir_path)
    else:
        # If the directory doesn't exist, create it
        os.mkdir(dir_path)
        print(f"Created directory: {dir_path}")

    dict_new_old_label_id = {}

    for lines in open("data/combine_label.txt"):
        line = lines.strip().split("\t")
        old_label=line[0]
        new_label=line[1]
        if new_label not in dict_new_old_label_id.keys():
            dict_new_old_label_id[new_label]=[]
        dict_new_old_label_id[new_label].append(old_label)

    dict_old_label_id_name={}
    for lines in open("data/label_count_id.csv"):
        line=lines.strip().split("\t")
        label_name=line[0]
        label_id=line[2]
        dict_old_label_id_name[label_id]=label_name

    test_steps=100
    prob=[]
    softmax_fun = nn.Softmax(dim=1)
    model.eval()

    for module in model.modules():
        if module.__class__.__name__.startswith('Dropout'):
            module.train()

    with torch.no_grad():
        for test_iter in range(test_steps):
            for sid, (sequence, contig) in enumerate(loader):
                sequence = sequence.to(device)
                test_output1 = model(sequence)
                test_output = torch.softmax(test_output1, dim=1)
                _, predictions = torch.max(softmax_fun(test_output1), dim=1)
                if sid == 0:
                    epoch_uncer = test_output
                    pred_y_all = predictions
                    contig_all=contig
                else:

                    epoch_uncer = torch.cat((epoch_uncer, test_output), 0)
                    pred_y_all = torch.cat((pred_y_all, predictions), 0)
                    contig_all= np.concatenate((contig_all,contig),0)

            pred_y_all = pred_y_all.cpu()

            np.savetxt(midfolder+"tmp/pre_" + str(test_iter) + ".txt", pred_y_all, fmt="%d")
            prob.append(epoch_uncer.cpu().numpy())

        predictive_variance = np.var(prob, axis=0)
        np.savetxt(midfolder+"tmp/var.txt", predictive_variance, fmt="%s")

        dict_contig_predictLable = {}

        for count, line in enumerate(open(f"{midfolder}test_protein_sentence.csv", 'rU')):
            pass
        count += 1
        sample_number = count - 1
        # print("sample number:",sample_number)

        for sample in range(sample_number):
            dict_contig_predictLable[sample] = []

        for i in range(test_steps):
            j = 0
            for lines in open(midfolder+"tmp/pre_" + str(i) + ".txt"):
                line = lines.strip().split("\t")
                dict_contig_predictLable[j].append(line[0])
                j = j + 1

        dict_predict_label = {}
        for k, v in dict_contig_predictLable.items():
            maxTimes_label = max(v, key=v.count)
            dict_predict_label[k] = maxTimes_label

        predict_label = list(dict_predict_label.values())

        var = np.loadtxt(midfolder+"tmp/var.txt")
        var_sample = []

        for l in range(len(predict_label)):
            column = int(predict_label[l])
            var_sample.append(var[l][column])

        file_final=open(midfolder+output,"w")
        file_final.write("contigs"+"\t"+"predict_label"+"\t"+"uncertainty"+"\n")
        for index in range(len(predict_label)):
            if var_sample[index]>var_cutoff:
                file_final.write(contig_all[index]+"\t"+"no predication"+"\t"+"NA"+"\n")
                continue
            else:
                new_id=str(predict_label[index])
                old_id=dict_new_old_label_id[new_id]
                if len(old_id)==1:
                    name=dict_old_label_id_name[old_id[0]]
                else:
                    name=""
                    for o in old_id[:-1]:
                        name=name+dict_old_label_id_name[o]+"/"
                    name=name+dict_old_label_id_name[old_id[-1]]


                file_final.write(contig_all[index]+"\t"+name+"\t"+str(var_sample[index])+"\n")
        file_final.close()

    os.system('rm -rf '+midfolder+"tmp/")

    return 1
