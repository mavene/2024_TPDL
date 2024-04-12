# import pandas

# # Saver for preprocessed dataframe
# def save_df(df, set):
#     df.to_csv(f'/work/preprocessed_{set}.csv', sep=',', encoding='utf-8', index=False)

# # Loader for preprocessed dataframe
# def load_df(path):
#     df = pandas.read_csv(path)
#     return df

# Saver/loader

import torch
import os
def save_model_ckpt(self, model, ):
    print("saving model checkpoint, itr:", itr)
    model_path = os.path.join(self.model_save_dir, '{}-.ckpt'.format(itr))

    torch.save(model.state_dict(), model_path)
    print('Saved model checkpoints into {}...'.format(self.model_save_dir))

# Restore the model 
# Note that itr is the value of the last checkpoint file
# The new starting epoch is (itr + 1) 
def restore_model(self, model, itr):
    print('Restore the trained models')
    model_path = os.path.join(self.model_save_dir, '{}-.ckpt'.format(itr))
       
    model.load_state_dict(torch.load(model_path))