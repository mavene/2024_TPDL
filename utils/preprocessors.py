import numpy

def fix_error_paths(row):
    row = row.replace("//", "/")
    return row

def str_to_array(row):
    ndarray = numpy.fromstring(
                row.replace('\n','')
                    .replace('[','')
                    .replace(']','')
                    .replace('  ',' '), 
                    sep=' ')
    return ndarray

def ohe_to_class(row):
    if numpy.sum(row) > 0:
        return numpy.argmax(row)
    else:
        return -100

def preprocess_pEffClassification(df):
    df.drop(columns=['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Edema', 'Pleural Other', 'Fracture', 'Support Devices'], axis=1, inplace=True)
    return df

def preprocess_cardioClassification(df):
    df.drop(columns=['Enlarged Cardiomediastinum', 'Pleural Effusion', 'Lung Opacity', 'Lung Lesion', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Edema', 'Pleural Other', 'Fracture', 'Support Devices'], axis=1, inplace=True)
    return df
