#this script writes a generic with the config parser function to ty and build batch run style scripts

#%%
import configparser

config = configparser.ConfigParser()
config['DataInfo'] = {'Directory':'D:/DAS/tmpData8m/',
                  'n_files':'5'}
config['ProcessingInfo'] = {'n_synthetic':'130',
                            'synthetic_spacing':'250',
                            'n_stack':'5',
                            'fs_target' : '256'
                            }
config['FFTInfo'] = {'N_fft':'512',
                     'n_overlap':'64',
                     'n_samp':'128'}
config['SaveInfo'] = {'plot_directory':'C:/Users/Calder/Ouputs/DASplots1/'}

with open('example.ini', 'w') as configfile:
  config.write(configfile)
# %%
