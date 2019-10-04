import ntpath
import os
from . import util, html
import time

def save_iamges(webpage, visuals, image_path, aspect_ratio=0.1, image_size=256):
    """

    :param webpage: the HTML class,
    :param visuals: Ordered Dict that stores (name, images)
    :param image_path:
    :param aspect_ratio:
    :param image_size:
    :return:
    """
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    image_paths, images_name, links = [], [], []

    for label, img_data in visuals.items():
        img = util.tensor2im(img_data)
        image_name = '{}_{}.png' .format(name, label)
        save_path = os.path.join(image_dir, image_name)
        util.save_image(img, save_path, aspect_ratio=aspect_ratio)
        image_paths.append(image_name)
        images_name.append(label)
        links.append(image_name)
    webpage.add_images(image_paths, images_name, links, image_size=image_size)

class Visualize:
    def __init__(self, opt):
        """

        :param opt:
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """
        self.opt = opt
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.port = opt.display_port
        self.saved = False
        if self.display_id > 0:  # connect to a visdom server given <display_port> and <display_server>
            import visdom
            self.ncols = opt.display_ncols
            self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, env=opt.display_env)
            if not self.vis.check_connection():
                self.create_visdom_connections()

        if self.use_html:  # create an HTML object at <checkpoints_dir>/web/; images will be saved under <checkpoints_dir>/web/images/
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory {}...' .format(self.web_dir))
            util.mkdirs([self.web_dir, self.img_dir])

        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss ({}) ================\n' .format(now))