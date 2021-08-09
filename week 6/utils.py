import re
import yaml
import logging
import tensorflow as tf
from csv import reader
import numpy as np
import os



def read_config_file(filepath):
    with open(filepath, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logging.error(exc)


def get_csv_header(file_path):
    with open(file_path, 'r') as read_obj:
        csv_reader = reader(read_obj)
        header = next(iter(csv_reader))
    return header


def replacer(string, char):
    pattern = char + '{2,}'
    string = re.sub(pattern, char, string) 
    return string

def header_cleaning(columns):
    columns = map(lambda x: x.lower(), columns)
    columns = map(lambda x: re.sub('[^\w]', '_', x), columns)
    columns = map(lambda x: x.strip('_'), columns)
    columns = map(lambda x: replacer(x,'_'), columns)
    return list(columns)

def validate(file_path,table_config):

    column_names = get_csv_header(file_path)
    #column_names = list(dataset.element_spec.keys())
    column_names = header_cleaning(column_names)
    column_names.sort()
    
    column_expected = table_config['columns']
    sort_expected_column=column_expected.copy()
    sort_expected_column.sort()
    
    if len(column_names) == len(column_expected) and column_names == sort_expected_column:
        print("validation passed")
        val_dataset = tf.data.experimental.make_csv_dataset(
            file_path,
            header=True,
            column_names=column_expected,
            batch_size=100,
            field_delim=table_config['inbound_delimiter'], 
            label_name=None,
            num_epochs=1,
            shuffle=False,
            ignore_errors=True,)
        
        return val_dataset

    else: 
        print("\ncolumn name and column length validation failed")
        mismatched_columns_file = list(set(column_names ).difference(sort_expected_column))
        if len(mismatched_columns_file)!=0:
            print("\nFollowing File columns are not in the YAML file",mismatched_columns_file) 
        missing_YAML_file = list(set(sort_expected_column).difference(column_names))
        if len(missing_YAML_file)!=0:
            print("\nFollowing YAML columns are not in the file uploaded",missing_YAML_file)
        
        return None


def to_string(x,delimiter):
    
    if isinstance(x, bytes):
        return x.decode() + delimiter
    else:
        return str(x) + delimiter


def count(dataset):
    s=0
    for batch in dataset:
        for key,values in batch.items(): 
            s+=len(values)
            break
    return s


def batch_to_write(batch):
    keys = []
    for n,(key,value) in enumerate(batch.items()):
        keys.append(key)

        if n != 0:
            values=np.vstack((values,value.numpy()))
        
        else:
            values = value.numpy()

    return keys,values.T 


def write_txt(txt_file,batch,delimiter):

    header, values = batch_to_write(batch)

    with open(txt_file,'a') as file:
        n_cols = len(header)
        delimiter=list(delimiter*(n_cols-1))
        delimiter.append('\n')
                
        if os.path.getsize(txt_file)==0:
            file.writelines(map(to_string,header,delimiter))
                    
        for row in values:
            line=list(map(to_string,list(row),delimiter))      
            file.writelines(line)
                    
    return n_cols
