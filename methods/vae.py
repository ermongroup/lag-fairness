import tensorflow as tf
import numpy as np
import pickle as pkl
import os
from scipy.stats import gaussian_kde

from tf_utils import logger, gpu_session, clear_dir

tfd = tf.contrib.distributions


class VariationalAutoEncoder(object):
    """
    Variational Auto Encoder
    """
    def __init__(self, encoder, decoder, datasets, optimizer, logdir):
        self.encoder = encoder
        self.decoder = decoder
        self.datasets = datasets
        self.optimizer = optimizer
        self.logdir = logdir
        self._create_datasets()
        self._create_loss()
        self._create_optimizer(encoder, decoder, optimizer)
        self._create_summary()
        self._create_evaluation(encoder, decoder)
        self._create_session(logdir)
        logger.configure(logdir, format_strs=['stdout', 'log'])

    def _create_datasets(self):
        datasets = self.datasets
        self.iterator = iterator = tf.data.Iterator.from_structure(
            output_types=datasets.train.output_types, output_shapes=datasets.train.output_shapes
        )
        self.train_init = iterator.make_initializer(datasets.train)
        self.test_init = iterator.make_initializer(datasets.test)

    def _create_loss(self):
        self.x = self.iterator.get_next()[0]
        z, logqzx = self.encoder.sample_and_log_prob(self.x)
        x_, logpxz, logpz = self.decoder.sample_and_log_prob(z, self.x)
        self.z = z

        self.encoder_loss = logqzx - logpz
        self.decoder_loss = -logpxz
        self.nll = tf.reduce_mean(self.decoder_loss)
        self.elbo = tf.reduce_mean(self.encoder_loss)
        self.loss = self.nll + self.elbo

    def _create_optimizer(self, encoder, decoder, optimizer):
        encoder_grads_and_vars = optimizer.compute_gradients(self.loss, encoder.vars)
        decoder_grads_and_vars = optimizer.compute_gradients(self.loss, decoder.vars)

        self.trainer = tf.group(optimizer.apply_gradients(encoder_grads_and_vars),
                                optimizer.apply_gradients(decoder_grads_and_vars))

    def _create_summary(self):
        with tf.name_scope('train'):
            self.train_summary = tf.summary.merge([
                tf.summary.scalar('elbo', self.elbo),
                tf.summary.scalar('nll', self.nll),
                tf.summary.scalar('loss', self.loss)
            ])

    def _create_evaluation(self, encoder, decoder):
        x = self.iterator.get_next()[0]
        self.z_mi = encoder.sample_and_log_prob(x)[0]
        self.log_q_z_x = encoder.sample_and_log_prob(x)[1]

    def _create_session(self, logdir):
        self.summary_writer = tf.summary.FileWriter(logdir=logdir)
        self.sess = gpu_session()
        self.saver = tf.train.Saver()
        self.logdir = logdir

    def _update_optimizer(self):
        pass

    def _debug(self):
        pass

    def _train(self):
        self._debug()
        self.sess.run([self.trainer])

    def _log(self, it):
        if it % 10 == 0:
            loss, nll, elbo = self.sess.run([self.loss, self.nll, self.elbo])
            logger.log("Iteration %d: loss %.4f nll %.4f elbo %.4f" % (it, loss, nll, elbo))
            self.summary_writer.add_summary(self.sess.run(self.train_summary), it)

    def train(self, num_epochs, num_iters=None):
        self.sess.run(tf.global_variables_initializer())
        it = 0
        for epoch in range(num_epochs):
            self.sess.run(self.train_init)
            self._update_optimizer()
            while True:
                try:
                    self._train()
                    it += 1
                    self._log(it)
                except tf.errors.OutOfRangeError:
                    break
                if num_iters and it > num_iters:
                    break
            if epoch % 100 == 1:
                print('Saving to: ', os.path.join(self.logdir, 'model/model.ckpt'))
                self.saver.save(sess=self.sess, save_path=os.path.join(self.logdir, 'model/model.ckpt'))
        self.sess.run(self.train_init)
        print('Saving to: ', os.path.join(self.logdir, 'model/model.ckpt'))
        self.saver.save(sess=self.sess, save_path=os.path.join(self.logdir, 'model/model.ckpt'))

    def test(self):
        print('Loading from ', os.path.join(self.logdir, 'model/model.ckpt'))
        self.saver.restore(sess=self.sess, save_path=os.path.join(self.logdir, 'model/model.ckpt'))
        self.sess.run(self.train_init)
        z_mis = []
        while True:
            try:
                z_mi = self.sess.run(self.z_mi)
                z_mis.append(z_mi)
            except tf.errors.OutOfRangeError:
                break
        z_mis = np.concatenate(z_mis, axis=0)
        kde = gaussian_kde(z_mis.transpose())

        self._evaluate_over_test_set(
            [self.elbo, self.nll, self.log_q_z_x],
            ['elbo', 'nll', 'logqzx']
        )

        self._estimate_mutual_information_continuous(self.z_mi, self.log_q_z_x)

    def _evaluate_over_test_set(self, keys, strs):
        self.sess.run(self.test_init)
        d = {'test_' + s: [] for s in strs}
        while True:
            try:
                ks = self.sess.run(keys)
                for i in range(len(keys)):
                    d['test_' + strs[i]].append(ks[i])
            except tf.errors.OutOfRangeError:
                break
        for k in d.keys():
            d[k] = np.mean(d[k])
        self._write_evaluation(d)

    def _evaluate_over_train_set(self, keys, strs):
        self.sess.run(self.train_init)
        d = {'train_' + s: [] for s in strs}
        while True:
            try:
                ks = self.sess.run(keys)
                # import ipdb; ipdb.set_trace()
                for i in range(len(keys)):
                    d['train_' + strs[i]].append(ks[i])
            except tf.errors.OutOfRangeError:
                break
        for k in d.keys():
            d[k] = np.mean(d[k])
        self._write_evaluation(d)

    def _estimate_mutual_information_continuous(self, z_var, qzx_var, label='qzx'):
        self.sess.run(self.train_init)
        zs, mis = [], []
        while True:
            try:
                z_mi = self.sess.run(z_var)
                zs.append(z_mi)
            except tf.errors.OutOfRangeError:
                break
        zs = np.concatenate(zs, axis=0)
        kde = gaussian_kde(zs.transpose())

        self.sess.run(self.test_init)
        while True:
            try:
                z, lqzx = self.sess.run([z_var, qzx_var])
                mi = lqzx - kde.logpdf(z.transpose())
                mis.append(np.mean(mi))
            except tf.errors.OutOfRangeError:
                break

        d = {'mi_' + label: np.mean(mis)}
        self._write_evaluation(d)

    def _estimate_mutual_information_discrete(self, z_var, qzx_var, qu0_var, qu1_var, y_var, label='quz'):
        self.sess.run(self.train_init)
        zs, mis = [], []
        mis0, mis1 = [], []
        zs0, zs1 = [], []
        while True:
            try:
                z_mi, y_mi = self.sess.run([z_var, y_var])
                zs.append(z_mi)
                zs0.append(z_mi[np.where(y_mi == 0)])
                zs1.append(z_mi[np.where(y_mi == 1)])
            except tf.errors.OutOfRangeError:
                break
        zs = np.mean(np.concatenate(zs, axis=0), axis=0)

        self.sess.run(self.test_init)
        while True:
            try:
                z, lqzx, qu0, qu1, y = self.sess.run([z_var, qzx_var, qu0_var, qu1_var, y_var])
                mi = lqzx - (z * np.log(zs) + (1.0 - z) * np.log(1.0 - zs))
                mis.extend(mi)
                mi = qu0 - (z * np.log(zs) + (1.0 - z) * np.log(1.0 - zs))
                mis0.extend(mi[np.where(y == 0)])
                mi = qu1 - (z * np.log(zs) + (1.0 - z) * np.log(1.0 - zs))
                mis1.extend(mi[np.where(y == 1)])
            except tf.errors.OutOfRangeError:
                break

        d = {
            'mi_' + label: np.mean(mis),
            'mi0_' + label: np.mean(mis0),
            'mi1_' + label: np.mean(mis1),
            'mi01_' + label: np.mean(mis0 + mis1)
        }
        self._write_evaluation(d)

    def _write_evaluation(self, d):
        logger.logkvs(d)
        logger.dumpkvs()
        try:
            with open(os.path.join(self.logdir, 'eval.pkl'), 'rb') as f:
                d_ = pkl.load(f)
        except FileNotFoundError:
            d_ = {}

        for k in d_.keys():
            if k not in d:
                d[k] = d_[k]

        with open(os.path.join(self.logdir, 'eval.pkl'), 'wb') as f:
            pkl.dump(d, f)



