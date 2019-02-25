import tensorflow as tf
import numpy as np
from scipy.stats import gaussian_kde
from tqdm import tqdm
from tf_utils import logger, gpu_session, clear_dir


from .vae import VariationalAutoEncoder

tfd = tf.contrib.distributions


class VariationalFairAutoEncoder(VariationalAutoEncoder):
    """
    Variational Fair Auto Encoder
    """
    def __init__(self, encoder, decoder, datasets, optimizer, logdir):
        super(VariationalFairAutoEncoder, self).__init__(encoder, decoder, datasets, optimizer, logdir)

    def _create_loss(self):
        self.x, self.u, self.y = self.iterator.get_next()
        z, logqzx = self.encoder.sample_and_log_prob(self.x, self.u)
        x_, logpxz, logpz = self.decoder.sample_and_log_prob(z, self.x, self.u)
        self.z = z

        self.encoder_loss = logqzx - logpz
        self.decoder_loss = -logpxz
        self.nll = tf.reduce_mean(self.decoder_loss)
        self.elbo = tf.reduce_mean(self.encoder_loss)
        self.loss = self.nll + self.elbo

    def _create_evaluation(self, encoder, decoder):
        x, u, y = self.iterator.get_next()
        self.z_mi = encoder.sample_and_log_prob(x, u)[0]
        self.log_q_z_x = encoder.sample_and_log_prob(x, u)[1]

    def _train(self):
        self._debug()
        self.sess.run([self.trainer])

    def _log(self, it):
        if it % 1000 == 0:
            loss, nll, elbo = self.sess.run([self.loss, self.nll, self.elbo])
            logger.log("Iteration %d: loss %.4f nll %.4f elbo %.4f" % (it, loss, nll, elbo))
            self.summary_writer.add_summary(self.sess.run(self.train_summary), it)

    def train(self, num_epochs):
        self.sess.run(tf.global_variables_initializer())
        it = 0
        for epoch in tqdm(range(num_epochs)):
            self.sess.run(self.train_init)
            self._update_optimizer()
            while True:
                try:
                    self._train()
                    it += 1
                    self._log(it)
                except tf.errors.OutOfRangeError:
                    break
        self.saver.save(sess=self.sess, save_path=self.logdir)

    def test(self):
        self.sess.run(self.train_init)
        self.saver.restore(sess=self.sess, save_path=self.logdir)
        z_mis = []
        while True:
            try:
                z_mi = self.sess.run(self.z_mi)
                z_mis.append(z_mi)
            except tf.errors.OutOfRangeError:
                break
        z_mis = np.concatenate(z_mis, axis=0)
        kde = gaussian_kde(z_mis.transpose())

        self.sess.run(self.test_init)
        elbo, nll, mis = [], [], []
        while True:
            try:
                e, n, z, lqzx = self.sess.run([self.elbo, self.nll, self.z[0], self.log_q_z_x])
                mi = lqzx - kde.logpdf(z.transpose())
                elbo.append(e)
                nll.append(n)
                mis.append(np.mean(mi))
            except tf.errors.OutOfRangeError:
                break
        logger.log('Test: all %.4f nll %.4f elbo %.4f mi %.4f' % (float(np.mean(nll) + np.mean(elbo)),
                                                                  float(np.mean(nll)), float(np.mean(elbo)),
                                                                  float(np.mean(mis))))


class VariationalFairClassifier(VariationalAutoEncoder):
    """
    Variational Fair Auto Encoder
    """
    def __init__(self, encoder, decoder, datasets, optimizer, logdir):
        super(VariationalFairClassifier, self).__init__(encoder, decoder, datasets, optimizer, logdir)

    def _create_loss(self):
        self.x, self.u, self.y = self.iterator.get_next()
        z, logqzx, logqyz = self.encoder.sample_and_log_prob(self.x, self.u, self.y)
        x_, logpxz, logpz = self.decoder.sample_and_log_prob(z, self.x, self.u, self.y)
        self.z = z

        self.error = tf.reduce_mean(-logqyz)
        self.encoder_loss = logqzx[0] + logqzx[1] - logpz[0] - logpz[1]
        self.decoder_loss = -logpxz
        self.nll = tf.reduce_mean(self.decoder_loss)
        self.elbo = tf.reduce_mean(self.encoder_loss)
        self.loss = self.nll + self.elbo + self.error

    def _create_evaluation(self, encoder, decoder):
        x, u, y = self.iterator.get_next()
        self.z_mi = encoder.sample_and_log_prob(x, u, y)[0][0]
        self.log_q_z_x = encoder.sample_and_log_prob(x, u, y)[1][0]

    def _log(self, it):
        if it % 1000 == 0:
            loss, nll, elbo, error = self.sess.run([self.loss, self.nll, self.elbo, self.error])
            logger.log("Iteration %d: loss %.4f nll %.4f elbo %.4f error %.4f" % (it, loss, nll, elbo, error))
