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

# Saves the model
def save_model_ckpt(self, model, epoch, optimizer, loss):
    print("Saving model checkpoint on epoch %d:", epoch)
    model_path = os.path.join(self.model_save_dir, '{}-.ckpt'.format(epoch))

    torch.save({
        'epoch' : epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item()
    }, model_path)

# Restore the model
def restore_model(self, model, epoch, optimizer, loss):
    print('Restore the trained models')
    model_path = os.path.join(self.model_save_dir, '{}-.ckpt'.format(epoch))
       
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    #optimizer.load_state_dict(torch.load(model_path)['optimizer_state_dict'])
    #loss.load_state_dict(torch.load(model_path)['loss'])
    #loss.load_state_dict(torch.load(model_path)['epoch'])

    model.eval()