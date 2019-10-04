from options.train_options import TrainOptions
from data import create_dataset
from models import create_model

if __name__ == '__main__':
    opt = TrainOptions().parse()
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print('The number of training images = {}'.format(dataset_size))
    model = create_model(opt)
    model.setup(opt
    visualizer = Visualizer(opt)
