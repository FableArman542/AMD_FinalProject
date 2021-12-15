from os import read
import numpy as np
import csv
import Orange as DM



def csv_to_tab(filename, new_filename):
    f = open(filename, 'rt')
    reader = list(csv.reader(f))

    class_row = [""]*len(reader[0])
    class_row[reader[0].index("class")] = "class"

    domain_row = ["discrete"]*len(reader[0])


    reader.insert(1, domain_row)
    reader.insert(2, class_row)

    csv_lines = []
    for line in reader:
        csv_lines.append("\t".join(line))

    with open(new_filename, mode='w') as file:
        file.write("\n".join(csv_lines))


def load(filename):
    try:
        dataset = DM.data.Table( filename )
    except:
        print("Error Loading File")
        exit()
    return dataset
    

csv_to_tab("dataset_long_name_ORIGINAL.csv", "dataset_long_name_EXPORTED.tab")