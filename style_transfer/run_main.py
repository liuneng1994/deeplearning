import tensorflow as tf
import argparse
import numpy as np
from style_transfer import utils
import style_transfer.style_transfer_model as style_transfer
import os

from style_transfer import vgg19


def parse_args():
    desc = "Style Transfer"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--model_path', type=str, default='C:\\Users\liuneng\Documents\Zapya\Misc',
                        help='The directory where the pre-trained model was saved')
    parser.add_argument('--content', type=str, default='C:\\Users\liuneng\Documents\Zapya\Photo\saber.jpg',
                        help='File path of content image (notation in the paper : p)')
    parser.add_argument('--style', type=str, default='F:\下载\\fa527ff61e2ccfdeac233a1c3a14b297.jpg',
                        help='File path of style image (notation in the paper : a)')
    parser.add_argument('--output', type=str, default='result.jpg', help='File path of output image')

    parser.add_argument('--loss_ratio', type=float, default=1e-1, help='Weight of content-loss relative to style-loss')

    parser.add_argument('--content_layers', nargs='+', type=str, default=['conv4_2'],
                        help='VGG19 layers used for content loss')
    parser.add_argument('--style_layers', nargs='+', type=str,
                        default=['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'],
                        help='VGG19 layers used for style loss')

    parser.add_argument('--content_layer_weights', nargs='+', type=float, default=[1.0],
                        help='Content loss for each content is multiplied by corresponding weight')
    parser.add_argument('--style_layer_weights', nargs='+', type=float, default=[.2, .2, .2, .2, .2],
                        help='Style loss for each content is multiplied by corresponding weight')

    parser.add_argument('--initial_type', type=str, default='content', choices=['random', 'content', 'style'],
                        help='The initial image for optimization (notation in the paper : x)')
    parser.add_argument('--max_size', type=int, default=512, help='The maximum width or height of input images')

    parser.add_argument('--steps', type=int, default=1000, help='The number of iterations to run')

    parser.add_argument('--log_interval', type=int, default=100, help='step interval of log and generated image')

    parser.add_argument('--image_log_dir', type=str, default='log', help='generated image log path')

    parser.add_argument('--learning_rate', type=float, default=0.00001, help='model learning rate')

    return check_args(parser.parse_args())


def check_args(args):
    try:
        assert len(args.content_layers) == len(args.content_layer_weights)
    except:
        print ('content layer info and weight info must be matched')
        return None
    try:
        assert len(args.style_layers) == len(args.style_layer_weights)
    except:
        print('style layer info and weight info must be matched')
        return None

    try:
        assert args.max_size > 100
    except:
        print ('Too small size')
        return None

    model_file_path = args.model_path + '/' + vgg19.MODEL_FILE_NAME
    try:
        assert os.path.exists(model_file_path)
    except:
        print ('There is no %s'%model_file_path)
        return None

    try:
        size_in_KB = os.path.getsize(model_file_path)
        assert abs(size_in_KB - 534904783) < 10
    except:
        print('check file size of \'imagenet-vgg-verydeep-19.mat\'')
        print('there are some files with the same name')
        print('pre_trained_model used here can be downloaded from bellow')
        print('http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat')
        return None

    try:
        assert os.path.exists(args.content)
    except:
        print('There is no %s'%args.content)
        return None

    try:
        assert os.path.exists(args.style)
    except:
        print('There is no %s' % args.style)
        return None

    return args


def main():
    args = parse_args()
    if args is None:
        exit()

    content_image = utils.load_image(args.content, max_size=args.max_size)
    style_image = utils.load_image(args.style, shape=[content_image.shape[1],content_image.shape[0]])

    init_image = None
    if args.initial_type == 'content':
        init_image = content_image
    elif args.initial_type == 'style':
        init_image = style_image
    elif args.initial_type == 'random':
        init_image = np.random.normal(size=content_image.shape, scale=np.std(content_image))

    assert init_image is not None
    # create a map for content layers info
    CONTENT_LAYERS = {}
    for layer, weight in zip(args.content_layers, args.content_layer_weights):
        CONTENT_LAYERS[layer] = weight

    # create a map for style layers info
    STYLE_LAYERS = {}
    for layer, weight in zip(args.style_layers, args.style_layer_weights):
        STYLE_LAYERS[layer] = weight

    with tf.Session() as sess:
        content_image = np.expand_dims(content_image, 0)
        style_image = np.expand_dims(style_image, 0)
        init_image = np.expand_dims(init_image, 0)
        model = style_transfer.StyleTransfer(style_transfer.Options(model_path=args.model_path,
                                                                    content_image=content_image,
                                                                    style_image=style_image,
                                                                    init_image=init_image,
                                                                    content_layers=CONTENT_LAYERS,
                                                                    learning_rate=args.learning_rate,
                                                                    log_interval=args.log_interval,
                                                                    image_log_dir=args.image_log_dir,
                                                                    loss_ratio=args.loss_ratio,
                                                                    steps=args.steps,
                                                                    style_layers=STYLE_LAYERS))
        final_image = model.generate(sess)
        utils.save_image(final_image, args.output)

