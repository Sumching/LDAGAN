import sys
import os
import tensorflow as tf
import urllib
import tarfile
import numpy as np
sys.path.append('./utils/')

def _download_inception_if_needed():
    filepath = os.path.join(MODEL_DIR, INCEPTION_GRAPH_NAME)
    if os.path.exists(filepath):
        return
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    filename = INCEPTION_URL.split('/')[-1]
    filepath = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(filepath):
      def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %s %.1f%%' % (
            filename, float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()
      filepath, _ = urllib.urlretrieve(INCEPTION_URL, filepath, _progress)
      print()
      statinfo = os.stat(filepath)
      print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
      tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)

tfgan = tf.contrib.gan

MODEL_DIR = './inception/'
INCEPTION_GRAPH_NAME = 'inceptionv1_for_inception_score.pb'
INCEPTION_INPUT = 'Mul:0'
INCEPTION_OUTPUT = 'logits:0'
INCEPTION_FINAL_POOL = 'pool_3:0'
INCEPTION_DEFAULT_IMAGE_SIZE = 299
INCEPTION_SHAPE = [INCEPTION_DEFAULT_IMAGE_SIZE, INCEPTION_DEFAULT_IMAGE_SIZE]

INCEPTION_URL = 'http://download.tensorflow.org/models/frozen_inception_v1_2015_12_05.tar.gz'
#INCEPTION_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

#### some parts are taken from https://github.com/igul222/improved_wgan_training
class Inception():
    def __init__(self):
        _download_inception_if_needed()
        self.batch_size = 500
        self.images_ph = tf.placeholder(tf.float32, shape=(None, None, None, None))
        self._preprocess_images()
        self.inception_input = tf.placeholder(tf.float32, shape=(self.batch_size, 32, 32, 3))
        #self.inception_input = (self.inception_input1 - 127.5) / 127.5
        self._import_inception_graph()
        self.fake_activations_ph = tf.placeholder(tf.float32, shape=(None, None))
        self.real_activations_ph = tf.placeholder(tf.float32, shape=(None, None))
        self.inception_scores = tfgan.eval.classifier_score_from_logits(self.fake_activations_ph)
        self.frechet_distance = tfgan.eval.frechet_classifier_distance_from_activations(
            self.real_activations_ph, self.fake_activations_ph)

    def _preprocess_images(self):
        resized_images = tf.image.resize_bilinear(self.images_ph,
                                                  INCEPTION_SHAPE)
        self.inception_input_temp = (resized_images - 127.5) / 127.5

    def _import_inception_graph(self):
        with tf.gfile.FastGFile(os.path.join(MODEL_DIR, INCEPTION_GRAPH_NAME),
                                'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            input_map = {"ExpandDims:0": self.inception_input}
            #input_map = {INCEPTION_INPUT: self.inception_input}
            output_tensors = [INCEPTION_FINAL_POOL, "softmax/logits/MatMul", INCEPTION_OUTPUT]
            inception_output = tf.import_graph_def(
                graph_def, input_map, output_tensors, name='inception')

            self.inception_features = tf.squeeze(inception_output[0])
            #print(self.inception_features)
            # https://github.com/openai/improved-gan/issues/29
            # Fix this for the future. In practice it doesn't matter much.
            w = inception_output[1].inputs[1]
            logits = tf.matmul(self.inception_features, w)
            self.logits = tf.nn.softmax(logits)
            self.logits1 = inception_output[2][:, :1001]
            self.preds = tf.exp(self.logits1) / tf.reduce_sum(tf.exp(self.logits1), 1, keepdims=True)
    def _IS(self, x, sess):
        num_data = x.shape[0] - x.shape[0] % (self.batch_size)
        batches = make_batches(num_data, self.batch_size)
        activation = []
        logist = []
        k_step = 0
        t_step = 0
        for batch_idx, (batch_start, batch_end) in enumerate(batches):
            batch_size = batch_end - batch_start

            x_batch = x[batch_start:batch_end]
            img = sess.run(self.inception_input_temp,{self.images_ph:x_batch})
            a, b = sess.run([self.inception_features, self.preds], {self.inception_input:img})
            activation.append(a)
            logist.append(b)
            scores = []
            if k_step % 10 == 9:
                activation = np.vstack(activation)
                logist = np.vstack(logist)
                print(logist.shape)

                score = sess.run(self.inception_scores, {self.fake_activations_ph:logist})


                print(np.mean(score))

                t_step += 1
                activation = []
                logist = []
            k_step += 1

    def _fid(self, x, r_mu, r_sigma, sess):
        num_data = x.shape[0]
        batches = make_batches(num_data, self.batch_size)
        f_act = []
        logist = []
        k_step = 0
        t_step = 0
        t_score = []
        for batch_idx, (batch_start, batch_end) in enumerate(batches):
            batch_size = batch_end - batch_start

            x_batch = x[batch_start:batch_end]

            #img = sess.run(self.inception_input_temp, {self.images_ph: x_batch})
            a, b = sess.run([self.inception_features, self.logits], {self.inception_input:x_batch})
            f_act.append(a)
            logist.append(b)
            if k_step % 10 == 9:
                logist = np.vstack(logist)

                part = logist
                kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
                kl = np.mean(np.sum(kl, 1))
                #score = sess.run(self.inception_scores, {self.fake_activations_ph:logist})
                score = np.exp(kl)
                t_score.append(np.mean(score))
                print('IS: %.4f' % np.mean(score))
                logist = []
            if t_step % 100 == 99:
                f_act = np.vstack(f_act)
                f_mu = np.mean(f_act, axis=0)
                f_sigma = np.cov(f_act, rowvar=False)
                fid = calculate_frechet_distance(f_mu, f_sigma, r_mu, r_sigma)

                print('FID: %.4f' % fid)

                print('IS_mean: %.4f' % np.mean(t_score), 'IS_std: %.4f'% np.std(t_score))
                f_act = []
            t_step +=1
            k_step += 1

    def sta(self, x, sess):
        num_data = x.shape[0] - x.shape[0] % (self.batch_size)
        batches = make_batches(num_data, self.batch_size)
        act = []
        for batch_idx, (batch_start, batch_end) in enumerate(batches):
            batch_size = batch_end - batch_start

            x_batch = x[batch_start:batch_end]
            a = sess.run(self.inception_features, {self.inception_input: x_batch})
            act.append(a)

        act = np.vstack(act)
        print(act.shape)
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        np.savez('celeba_sta', mu=mu, sigma=sigma)
def make_batches(size, batch_size):
    '''Returns a list of batch indices (tuples of indices).
    '''
    return [(i, min(size, i + batch_size)) for i in range(0, size, batch_size)]

from scipy import linalg
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


if __name__ == "__main__":
    import argparse

    #argument
    parser = argparse.ArgumentParser(
    description='Calculate Inception Score and Fréchet Inception Distance.'
    )
    parser.add_argument('--input', type=str, help='samples with npy format')
    parser.add_argument('--stats', type=str, help='pre-calculated statistics')
    args = parser.parse_args()

    r_cifar_act = np.load(args.stats)
    r_mu = r_cifar_act['mu']
    r_sigma = r_cifar_act['sigma']
    x_train = np.load(args.input)
    
    inception = Inception()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                            log_device_placement=False)) as sess:
        inception._fid(x_train, r_mu, r_sigma, sess)
        #inception._IS(x_train, sess)
        #inception.sta(x_train, sess)
