import tensorflow as tf
import numpy as np
import style_transfer.utils as utils
import os
from style_transfer import vgg19


class Options:
    """
    Model config
    """

    def __init__(self, model_path, content_image, init_image, style_image, content_layers, style_layers,
                 loss_ratio, steps, log_interval, image_log_dir=None):
        # path of saving generated image log
        self.image_log_dir = image_log_dir
        # interval of saving image log
        self.log_interval = log_interval
        # max training step num
        self.steps = steps
        # vgg19 model path
        self.vgg_model_path = model_path
        # ndarray of content image
        self.content_image = content_image
        # ndarray of init image
        self.init_image = init_image
        # ndarray of style image
        self.style_image = style_image
        # layer weights of content loss
        self.content_layers = content_layers
        # layer weights of style layer
        self.style_layers = style_layers
        # content loss ratio,style loss ratio always is 1
        self.loss_ratio = loss_ratio


class StyleTransfer:
    def __init__(self, option: Options):
        self.option = option
        self.vgg_model = vgg19.VGG19(os.path.join(self.option.vgg_model_path, vgg19.MODEL_FILE_NAME))
        self.content_image = self.vgg_model.preprocess(self.option.content_image)
        self.init_image = self.vgg_model.preprocess(self.option.init_image)
        self.style_image = self.vgg_model.preprocess(self.option.style_image)
        self._build_graph()
        pass

    def _build_graph(self):
        # image input
        self.content = tf.placeholder(tf.float32, shape=self.content_image.shape, name='content')
        self.style = tf.placeholder(tf.float32, shape=self.content_image.shape, name='style')

        # generated image
        self.init = tf.get_variable("generated_image", shape=self.init_image.shape, dtype=tf.float32)

        content_layers = self.vgg_model.feed_forward(self.content, scope='content')
        self.content_features = {}
        for layer in self.option.content_layers:
            self.content_features[layer] = content_layers[layer]

        style_layers = self.vgg_model.feed_forward(self.style, scope='style')
        self.style_features = {}
        for layer in self.option.style_layers:
            self.style_features[layer] = style_layers[layer]

        gen_features = self.vgg_model.feed_forward(self.init, scope='mixed')
        self._total_loss(gen_features)

    def generate(self, sess):
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.option.learning_rate)
        # step = optimizer.minimize(self.L_total)
        num_step = self.option.steps
        log_interval = self.option.log_interval
        image_log_dir = self.option.image_log_dir
        gen_image = self.init
        sess.run(tf.global_variables_initializer())
        sess.run(tf.assign(self.init,self.init_image))
        global step
        step = 0
        def callback(total_loss,content_loss, style_loss,image):
            global step
            print('step : %4d, ' % step,
                  'L_total : %g, L_content : %g, L_style : %g' % (total_loss, content_loss, style_loss))
            if step % log_interval == 0:
                if image_log_dir is not None:
                    if not os.path.exists(image_log_dir):
                        os.mkdir(image_log_dir)
                    filename = 'step_%d.jpg' % step
                    full_path = os.path.join(image_log_dir, filename)
                    image = np.clip(self.vgg_model.undo_preprocess(image), 0, 255)
                    image = image.reshape(image.shape[1:])
                    utils.save_image(image, path=full_path)
            step += 1

        optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.L_total, method='L-BFGS-B', options={'maxiter': num_step})
        optimizer.minimize(sess,feed_dict=self._get_feed_dict(),fetches=[self.L_total,self.L_content,self.L_style,self.init], loss_callback=callback)

        final_image = self.init.eval(sess)
        final_image = np.clip(self.vgg_model.undo_preprocess(final_image), 0, 255)
        final_image = final_image.reshape(final_image.shape[1:])
        return final_image


    def _get_feed_dict(self):
        return {self.content: self.content_image, self.style: self.style_image}


    def _content_loss(self, gen_features):
        content_loss = 0
        for layer_name in self.content_features:
            C = self.content_features[layer_name]
            G = gen_features[layer_name]
            weight = self.option.content_layers[layer_name]
            _, h, w, c = G.get_shape()
            N = h.value * w.value
            M = c.value
            content_loss += weight * tf.reduce_sum(tf.pow(G - C, 2)) / (2 * np.sqrt(M) * np.sqrt(N))
        return content_loss


    def _style_loss(self, gen_features):
        style_loss = 0
        for layer_name in self.style_features:
            S = self._gram_matrix(self.style_features[layer_name])
            G = gen_features[layer_name]
            weight = self.option.style_layers[layer_name]
            _, h, w, c = G.get_shape()
            N = h.value * w.value
            M = c.value
            G = self._gram_matrix(G)
            style_loss += weight * (1. / (4 * M ** 2 * N ** 2)) * tf.reduce_sum(tf.pow(G - S, 2))
        return style_loss


    def _total_loss(self, gen_features):
        self.L_content = self._content_loss(gen_features)
        self.L_style = self._style_loss(gen_features)
        alpha = self.option.loss_ratio
        beta = 1
        self.L_total = alpha * self.L_content + beta * self.L_style
        return self.L_content, self.L_style, self.L_total


    def _gram_matrix(self, x):
        shape = x.get_shape()
        channels = shape[3]
        matrix = tf.reshape(x, (-1, channels))
        gram = tf.matmul(tf.transpose(matrix), matrix)
        return gram
