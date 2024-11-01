#this script experiments with the config parser function to ty and build batch run style scripts

#%%
import configparser

config = configparser.ConfigParser()
config['Data'] = {'Directory':'C:/Users/calderr/Downloads/tempData/',
                  'Start_DT':'2024-03-01 02:02:02',
                  'End_DT': '2024-04-01 02:02:02',
                  'FilesPerChunk': '5'}
config['ProcessingInfo'] = {'DetectorSpacing_ch':'250',
                            'N_ChannelStack':'10',
                            'fsTarget':'256',
                            'NFFT' : '512'}

with open('example.ini', 'w') as configfile:
  config.write(configfile)
# %%
