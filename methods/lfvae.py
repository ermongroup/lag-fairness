import tensorflow as tf
import numpy as np
from tqdm import tqdm
from tf_utils import logger
import pickle as pkl
import os
# from .vfae import VariationalFairClassifier, VariationalFairAutoEncoder
from .vae import VariationalAutoEncoder
from utils import demographic_parity, equalized_odds, equalizied_opportunity, accuracy
from scipy.stats import gaussian_kde

tfd = tf.contrib.distributions


class LagrangianFairTransferableAutoEncoder(VariationalAutoEncoder):
    """
    mi: mutual information term
    non-lagrangian case:
    e1: vae term, consistency constraints (if set to 0, it is autoencoder)
    e2: adversartial term (if set to 0, we only use vae)
    e3: upper bound to I(x, u; z) (if we set to 0,
    """
    def __init__(self, encoder, decoder, datasets, optimizer, logdir, pu, mi, e1, e2, e3, e4, e5, lagrangian, disc):
        self.pu = pu
        self.lagrangian = lagrangian
        self.disc = disc
        self.e1, self.e2, self.e3, self.e4, self.e5, self.mi = e1, e2, e3, e4, e5, mi
        self.global_step=None
        super(LagrangianFairTransferableAutoEncoder, self).__init__(encoder, decoder, datasets, optimizer, logdir)

    def _create_loss(self):
        self.x, self.u, self.y = self.iterator.get_next()
        z, u_, logqzx, logqu_, logqu, qy, logqu0, logqu1 = self.encoder.sample_and_log_prob(self.x, self.u, self.y)
        x_, logpxz, logpz = self.decoder.sample_and_log_prob(z, self.x, self.u)
        self.x_ = x_
        self.z = z
        self.logpz = logpz
        logpuz = self.pu.log_prob(self.u)
        self.logqud = [logqu, logqu_, logpuz]
        self.logqzx = logqzx
        self.qu = tfd.Bernoulli(probs=tf.reduce_mean(tf.cast(u_, tf.float32), axis=0))
        self.um = tf.reduce_mean(tf.cast(u_, tf.float32) - tf.cast(self.u, tf.float32))
        logquz = self.qu.log_prob(u_)
        self.vae = vae = tf.reduce_mean(logqzx - logpxz - logpz)  # vae loss (consistency constraints)
        self.elbo = elbo = tf.reduce_mean(logqzx - logpz)
        self.lld = lld = tf.reduce_mean(-logpxz)  # reconstruction error
        self.mi_z_u = mi_z_u = tf.reduce_mean(logqu - logpuz)  # estimated mutual information upper bound
        self.mi_z_u0 = self.mi_z_u0t = tf.boolean_mask(logqu0 - logpuz, tf.equal(self.y, 0))
        self.mi_z_u1 = self.mi_z_u1t = tf.boolean_mask(logqu1 - logpuz, tf.equal(self.y, 1))

        self.mi_z_u0, self.mi_z_u1 = tf.reduce_mean(self.mi_z_u0), tf.reduce_mean(self.mi_z_u1)
        self.logqu = tf.reduce_mean(logqu)  # adversary objective
        self.logqu0 = tf.reduce_mean(logqu0)
        self.logqu1 = tf.reduce_mean(logqu1)
        self.logqy = tf.reduce_mean(qy.log_prob(self.y))  # classification error
        self.y_ = qy.logits
        self.u_ = u_

        def _get_lambda(name, e):
            if e > 0:
                return tf.get_variable(name, shape=[], dtype=tf.float32,
                                       initializer=tf.constant_initializer(1.0), trainable=True)
            else:
                return tf.get_variable(name, shape=[], dtype=tf.float32,
                                       initializer=tf.constant_initializer(0.0), trainable=False)

        if self.lagrangian:
            self.l1 = _get_lambda('lambda1', self.e1)
            self.l2 = _get_lambda('lambda2', self.e2)
            self.l3 = _get_lambda('lambda3', self.e3)
            self.l4 = _get_lambda('lambda4', self.e4)
            self.l5 = _get_lambda('lambda5', self.e5)
            self.loss = self.mi * lld + self.l1 * elbo + self.l2 * mi_z_u + self.l3 * vae + self.l4 * (self.mi_z_u0 + self.mi_z_u1) / 2 + self.l5 * self.mi_z_u1\
                        - self.l1 * self.e1 - self.l2 * self.e2 - self.l3 * self.e3 - self.l4 * self.e4 - self.l5 * self.e5
        else:
            self.l1 = tf.get_variable('lambda1', shape=[], dtype=tf.float32, initializer=tf.constant_initializer(self.e1))
            self.l2 = tf.get_variable('lambda2', shape=[], dtype=tf.float32, initializer=tf.constant_initializer(self.e2))
            self.l3 = tf.get_variable('lambda3', shape=[], dtype=tf.float32, initializer=tf.constant_initializer(self.e3))
            self.l4 = tf.get_variable('lambda4', shape=[], dtype=tf.float32, initializer=tf.constant_initializer(self.e4))
            self.l5 = tf.get_variable('lambda5', shape=[], dtype=tf.float32, initializer=tf.constant_initializer(self.e5))
            self.loss = self.mi * lld + self.l1 * elbo + self.l2 * mi_z_u + self.l3 * vae

    def _create_optimizer(self, encoder, decoder, optimizer):
        encoder_grads_and_vars = optimizer.compute_gradients(self.loss, encoder.vars)
        decoder_grads_and_vars = optimizer.compute_gradients(self.loss, decoder.vars)
        disc_grads_and_vars = optimizer.compute_gradients(-self.logqu - self.logqu0 - self.logqu1, encoder.discriminate_vars)
        print(encoder.discriminate_vars)

        global_step = self.global_step
        self.trainer = tf.group(optimizer.apply_gradients(encoder_grads_and_vars, global_step=global_step),
                                optimizer.apply_gradients(decoder_grads_and_vars))
        self.adversary = optimizer.apply_gradients(disc_grads_and_vars, global_step=global_step)
        if self.lagrangian:
            lambda_vars = [var for var in [self.l1, self.l2, self.l3, self.l4, self.l5] if var in tf.trainable_variables()]
            self.lambda_update = tf.train.RMSPropOptimizer(0.0001).minimize(-self.loss, var_list=lambda_vars)
            self.lambda_clip = tf.group(
                [tf.assign(var, tf.minimum(tf.maximum(var, 0.01), 100.0)) for var in lambda_vars]
            )

        self.classifier = optimizer.apply_gradients(
            optimizer.compute_gradients(-self.logqy, var_list=encoder.classify_vars))

    def _create_evaluation(self, encoder, decoder):
        pass

    def _create_summary(self):
        with tf.name_scope('train'):
            self.train_summary = tf.summary.merge([
                tf.summary.scalar('vae', self.vae),
                tf.summary.scalar('mi_z_u', self.mi_z_u),
                tf.summary.scalar('loss', self.loss),
                tf.summary.scalar('lambda1', self.l1),
                tf.summary.scalar('lambda2', self.l2)
            ])

    def _debug(self):
        pass
        # print(self.sess.run([self.mi_z_u, self.mi_z_u0, self.mi_z_u1]))
        # import ipdb; ipdb.set_trace()

    def _train(self):
        self._debug()
        if self.lagrangian:
            self.sess.run([self.trainer, self.adversary, self.lambda_update])
            self.sess.run(self.lambda_clip)
        else:
            self.sess.run([self.trainer, self.adversary])
        for i in range(self.disc-1):
            self.sess.run(self.adversary)

    def _log(self, it):
        if it % 1000 == 0:
            loss, lld, vae, mi_z_u, qu, um, elbo, l1, l2, l3, l4, l5, logpz, logqzx, mi_z_u0, mi_z_u1, _ = self.sess.run([
                self.loss, self.lld, self.vae, self.mi_z_u, self.logqu, self.um, self.elbo, self.l1, self.l2, self.l3,
                self.l4, self.l5, self.logpz, self.logqzx, self.mi_z_u0, self.mi_z_u1, self.trainer])
            logger.log('It %d: loss %.4f lld %.4f vae %.4f mi_z_u %.4f logqu %.4f um %.4f elbo %.4f l1 %.2f l2 %.2f l3 %.2f l4 %.2f l5 %.2f logpz %.2f logqzx %.2f mizu0 %.2f mizu1 %.2f' % (
                it, loss, lld, vae, mi_z_u, -qu, um, elbo, l1, l2, l3, l4, l5, np.mean(logpz), np.mean(logqzx), mi_z_u0, mi_z_u1
            ))
            self.summary_writer.add_summary(self.sess.run(self.train_summary), it)

    def learn_classifier(self):
        for _ in tqdm(range(100)):
            self.sess.run(self.train_init)
            while True:
                try:
                    self.sess.run(self.classifier)
                except tf.errors.OutOfRangeError:
                    break

    def evaluate_classifier(self, idx=0):
        self.sess.run(self.train_init)
        zs, ys = [], []
        while True:
            try:
                z, y = self.sess.run([self.z, self.y])
                zs.append(z)
                ys.append(y)
            except tf.errors.OutOfRangeError:
                break

        zs = np.concatenate(zs, axis=0)
        zsm = np.mean(zs, axis=0)
        zss = np.std(zs, axis=0)
        ys = np.concatenate(ys, axis=0)
        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression()
        lr.fit((zs - zsm) / zss, ys)
        ys_ = lr.predict((zs - zsm) / zss)

        from sklearn.metrics import roc_auc_score
        d = {
            'train_auc': roc_auc_score(ys, ys_),
            'train_acc': accuracy(ys, ys_)
        }

        self.sess.run(self.test_init)
        ys, ys_, us, zs = [], [], [], []
        while True:
            try:
                y, y_, u, z = self.sess.run([self.y, self.y_, self.u, self.z])
                ys.append(y)
                ys_.append(y_)
                us.append(u[:, idx])
                zs.append(z)
            except tf.errors.OutOfRangeError:
                break
        ys = np.reshape(np.concatenate(ys, axis=0), [-1])
        ys_ = np.reshape(np.concatenate(ys_, axis=0), [-1])
        us = np.reshape(np.concatenate(us, axis=0), [-1])
        zs = np.concatenate(zs, axis=0)
        ys_ = lr.predict((zs - zsm) / zss)
        with open(os.path.join(self.logdir, 'class.pkl'), 'wb') as f:
            pkl.dump([ys, ys_], f)

        d.update({
            'test_auc': roc_auc_score(ys, ys_),
            'test_dp': demographic_parity(ys_, us),
            'test_eodds': equalized_odds(ys, ys_, us),
            'test_eopp': equalizied_opportunity(ys, ys_, us),
            'test_acc': accuracy(ys, ys_)
        })

        self._write_evaluation(d)

    def test(self):
        d = {'mi': self.mi, 'e1': self.e1, 'e2': self.e2, 'e3': self.e3, 'e4': self.e4, 'e5': self.e5, 'disc': self.disc}
        self._write_evaluation(d)
        print('Loading from ', os.path.join(self.logdir, 'model/model.ckpt'))
        self.saver.restore(sess=self.sess, save_path=os.path.join(self.logdir, 'model/model.ckpt'))

        self._evaluate_over_train_set(
            [self.loss, self.vae, self.lld, self.mi_z_u, self.mi_z_u0, self.mi_z_u1, self.logqu, self.elbo],
            ['loss', 'vae', 'lld', 'mi_zu_bound', 'mi_zu0', 'mi_zu1', 'logqu', 'elbo']
        )
        self._evaluate_over_test_set(
            [self.loss, self.vae, self.lld, self.mi_z_u, self.mi_z_u0, self.mi_z_u1, self.logqu, self.elbo],
            ['loss', 'vae', 'lld', 'mi_zu_bound', 'mi_zu0', 'mi_zu1', 'logqu', 'elbo']
            # [self.loss, self.vae, self.lld, self.mi_z_u, self.logqu, self.elbo],
            # ['loss', 'vae', 'lld', 'mi_zu_bound', 'logqu', 'elbo']
        )

        self._estimate_conditional_mutual_information_continuous(self.z, self.logqzx, self.u, self.y)
        self._estimate_mutual_information_discrete(self.u_, self.logqu, self.logqu0, self.logqu1, self.y, label='quz')
        # self.learn_classifier()
        self.evaluate_classifier()

    def _estimate_conditional_mutual_information_continuous(self, z_var, qzx_var, u_var, y_var, label='zxiu'):
        self.sess.run(self.train_init)
        zs, mi0, mi1 = [[], []], [], []
        print(z_var, qzx_var, u_var)

        from sklearn.neighbors import KernelDensity
        while True:
            try:
                z_mi, u, qzx = self.sess.run([z_var, u_var, qzx_var])
                u = np.reshape(u, [-1])
                zs[0].append(z_mi[np.nonzero(1 - u)])
                zs[1].append(z_mi[np.nonzero(u)])
            except tf.errors.OutOfRangeError:
                break
        zs[0] = np.concatenate(zs[0], axis=0)
        zs[1] = np.concatenate(zs[1], axis=0)
        kde = [gaussian_kde(zs[0].transpose()), gaussian_kde(zs[1].transpose())]
        kde[0].set_bandwidth('silverman')
        kde[1].set_bandwidth('silverman')
        # kde_sk = [KernelDensity(), KernelDensity()]
        # kde_sk[0].fit(zs[0])
        # kde_sk[1].fit(zs[1])

        self.sess.run(self.test_init)
        while True:
            try:
                z, lqzx, u, y = self.sess.run([z_var, qzx_var, u_var, y_var])
                u = np.reshape(u, [-1])
                idx = [np.nonzero(1.0 - u), np.nonzero(u)]
                mi = lqzx[idx[0]] - kde[0].logpdf(z[idx[0]].transpose())
                y = np.reshape(y, [-1])
                for i in range(idx[0][0].__len__()):
                    if y[idx[0][0][i]] == 0:
                        mi0.append(mi[i])
                    else:
                        mi1.append(mi[i])
                mi = lqzx[idx[1]] - kde[1].logpdf(z[idx[1]].transpose())
                for i in range(idx[1][0].__len__()):
                    if y[idx[1][0][i]] == 0:
                        mi0.append(mi[i])
                    else:
                        mi1.append(mi[i])
            except tf.errors.OutOfRangeError:
                break

        d = {'mi_' + label: np.mean(mi0 + mi1)}
        self._write_evaluation(d)

    def _estimate_conditional_mutual_information_continuous_health(self, z_var, qzx_var, u_var, label='zxiu'):
        self.sess.run(self.train_init)
        zs, mis = [[] for i in range(0, 18)], []
        while True:
            try:
                z_mi, u = self.sess.run([z_var, u_var])
                for i in range(z_mi.shape[0]):
                    idx = int(np.sum(np.nonzero(u[i, :-1])) + 9 * u[i, -1])
                    zs[idx].append(z_mi[i:i+1])
            except tf.errors.OutOfRangeError:
                break
        zs = [np.concatenate(z, axis=0) for z in zs]
        kde = [gaussian_kde(z.transpose()) for z in zs]

        self.sess.run(self.test_init)
        while True:
            try:
                z, lqzx, u = self.sess.run([z_var, qzx_var, u_var])
                for i in range(z.shape[0]):
                    idx = int(np.sum(np.nonzero(u[i, :-1])) + 9 * u[i, -1])
                    mi = lqzx[i] - kde[idx].logpdf(z[i:i+1].transpose())
                    mis.append(mi)
            except tf.errors.OutOfRangeError:
                break
        d = {'mi_' + label: np.mean(mis)}
        self._write_evaluation(d)

    def _estimate_mutual_information_discrete_health(self, z_var, qzx_var, label='quz'):
        self.sess.run(self.train_init)
        zs, mis = [], []
        while True:
            try:
                z_mi = self.sess.run(z_var)
                zs.append(z_mi)
            except tf.errors.OutOfRangeError:
                break
        zs = np.mean(np.concatenate(zs, axis=0), axis=0, keepdims=True)
        zs = np.tile(zs, [64, 1])

        self.sess.run(self.test_init)
        while True:
            try:
                z, lqzx = self.sess.run([z_var, qzx_var])
                w = z.shape[0]
                mi = lqzx - (z[:, -1] * np.log(zs[:w, -1]) + (1.0 - z[:, -1]) * np.log(1.0 - zs[:w, -1]))
                mi -= np.log(zs[np.nonzero(z[:, :-1])])
                mis.append(np.mean(mi))
            except tf.errors.OutOfRangeError:
                break

        d = {'mi_' + label: np.mean(mis)}
        self._write_evaluation(d)

    def test_health(self):
        d = {'mi': self.mi, 'e1': self.e1, 'e2': self.e2, 'e3': self.e3, 'disc': self.disc}
        self._write_evaluation(d)
        self.saver.restore(sess=self.sess, save_path=os.path.join(self.logdir, 'model/model.ckpt'))

        self._evaluate_over_train_set(
            [self.loss, self.vae, self.lld, self.mi_z_u, self.logqu, self.elbo],
            ['loss', 'vae', 'lld', 'mi_zu_bound', 'logqu', 'elbo']
        )
        self._evaluate_over_test_set(
            [self.loss, self.vae, self.lld, self.mi_z_u, self.logqu, self.elbo],
            ['loss', 'vae', 'lld', 'mi_zu_bound', 'logqu', 'elbo']
        )

        self._estimate_conditional_mutual_information_continuous_health(self.z, self.logqzx, self.u)
        self._estimate_mutual_information_discrete_health(self.u_, self.logqu, label='quz')
        # self.learn_classifier()
        self.evaluate_classifier(idx=-1)

