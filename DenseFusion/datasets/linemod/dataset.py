from torch.utils.data import Dataset


class PoseDataset(Dataset):
    def __init__(self, mode, num_points, add_noise, dataset_root, noise_trans, refine):
        """
            :param mode: train/test/eval
            :param noise_trans:  range of noise to add for tranlation
            """

        self.dataset_root = dataset_root
        self.objlist = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
        self.mode = mode

        self.list_rgb = []
        self.list_depth = []
        self.list_label = []
        self.list_obj = []
        self.list_rank = []
        self.meta = {}
        self.pt = {}
        self.noise_trans = noise_trans
        self.refine = refine
        item_count = 0
        for item in self.objlist:
            if self.mode == 'train':
                input_file = open('{0}/data/{1}/train.txt'.format(self.dataset_root, '%02d' % item))
            else:
                input_file = open('{0}/data/{1}/test.txt'.format(self.dataset_root, '%02d' % item))

            while 1:
                item_count += 1
                input_line = input_file.readline()


