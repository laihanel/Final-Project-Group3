class Path(object):
    @staticmethod
    def db_dir(database):
        if database == 'ucf101':
            # folder that contains class labels
            root_dir = '/home/ubuntu/Deep-Learning/FinalProject/Data'

            # Save preprocess data into output_dir
            output_dir = '/home/ubuntu/Deep-Learning/FinalProject/ucf101'

            return root_dir, output_dir
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def model_dir():
        return '/path/to/Models/c3d-pretrained.pth'