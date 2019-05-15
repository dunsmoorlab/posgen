import sys
import os

def get_data_dir():
	#three cases: bash, windows, school mac
	if sys.platform == 'linux':
		return '/mnt/c/Users/ACH/Google Drive/FC_FMRI_DATA/'
	elif sys.platform == 'win32':
		return 'C:\\Users\\ACH\\Dropbox (LewPeaLab)\\Dunsmoor Lab\\posgen\\'
	else:
		return os.path.expanduser('~') + os.sep + 'Db_lpl/Dunsmoor Lab/posgen/'

data_dir = get_data_dir()