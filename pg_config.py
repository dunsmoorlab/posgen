import sys
import os
import seaborn as sns

#edit these on your own computer probably and then set it to git-ignore
def get_data_dir():
	#three cases: bash, windows, school mac
	if sys.platform == 'linux':
		return '/mnt/c/Users/ACH/Google Drive/FC_FMRI_DATA/'
	elif sys.platform == 'win32':
		return 'C:\\Users\\ACH\\Dropbox (LewPeaLab)\\Dunsmoor Lab\\posgen\\'
	else:
		return os.path.expanduser('~') + os.sep + 'Db_lpl/Dunsmoor Lab/posgen/'

data_dir = get_data_dir()

pospal = sns.color_palette('mako_r',5)
fearpal = sns.color_palette('hot_r',5)