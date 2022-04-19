import os

# %%------------------------------------------------------------------------------------------------------------------

OR_PATH = os.getcwd()
os.chdir("..")  # Change to the parent directory
PATH = os.getcwd()
DATA_DIR = os.getcwd() + os.path.sep + 'Data' + os.path.sep
PROCESS_DIR = os.getcwd() + os.path.sep + 'Process' + os.path.sep
MODEL_DIR = os.getcwd() + os.path.sep + 'Models' + os.path.sep
sep = os.path.sep

os.chdir(OR_PATH)  # Come back to the folder where the code resides , all files will be left on this directory


# %%
class Path(object):
    @staticmethod
    def db_dir(database):
        if database == 'ucf101':
            # folder that contains class labels
            root_dir = DATA_DIR

            # Save preprocess data into output_dir
            output_dir = PROCESS_DIR

            return root_dir, output_dir
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def model_dir():
        return MODEL_DIR + 'c3d-pretrained.pth'