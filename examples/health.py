import tensorflow as tf
import tf_utils as tu
import click
from data.health import create_health_dataset
from methods import LagrangianFairTransferableAutoEncoder

tfd = tf.contrib.distributions


class VariationalEncoder(object):
    def __init__(self, z_dim=30):
        def encoder_func(x):
            fc1 = tf.layers.dense(x, 100, activation=tf.nn.softplus)
            mean = tf.layers.dense(fc1, z_dim, activation=tf.identity)
            logstd = tf.layers.dense(fc1, z_dim, activation=tf.identity)
            return mean, tf.exp(logstd)

        def discriminate_func(z):
            fc1 = tf.layers.dense(z, 100, activation=tf.nn.softplus)
            sex_logits = tf.layers.dense(fc1, 1, activation=tf.identity)
            age_logits = tf.layers.dense(fc1, 9, activation=tf.identity)
            return sex_logits, age_logits

        def classify_func(z):
            fc1 = tf.layers.dense(z, 100, activation=tf.nn.softplus)
            logits = tf.layers.dense(fc1, 1, activation=tf.identity)
            return logits

        self.encoder = tf.make_template('encoder/x', lambda x: encoder_func(x))
        self.discriminate = tf.make_template('disc/u', lambda z: discriminate_func(z))
        self.discriminate_0 = tf.make_template('disc_0/u', lambda z: discriminate_func(z))
        self.discriminate_1 = tf.make_template('disc_1/u', lambda z: discriminate_func(z))
        self.classify = tf.make_template('classify/y', lambda z: classify_func(z))

    @property
    def vars(self):
        return [var for var in tf.global_variables() if 'encoder' in var.name]

    @property
    def discriminate_vars(self):
        return [var for var in tf.global_variables() if 'disc' in var.name]

    @property
    def classify_vars(self):
        return [var for var in tf.global_variables() if 'classify' in var.name]

    def sample_and_log_prob(self, x, u, y):
        loc1, scale1 = self.encoder(tf.concat([x, u], axis=1))
        qzx = tfd.MultivariateNormalDiag(loc=loc1, scale_diag=scale1)  # q(z_1 | x, u)
        z1 = qzx.sample()
        sex, age = self.discriminate(z1)
        qs = tfd.Bernoulli(logits=sex)
        qa = tfd.Multinomial(total_count=1., logits=age)
        u_ = tf.concat([qa.sample(), tf.cast(qs.sample(), tf.float32)], axis=1)

        sex0, age0 = self.discriminate_0(z1)
        qs0 = tfd.Bernoulli(logits=sex0)
        qa0 = tfd.Multinomial(total_count=1., logits=age0)
        u_0 = tf.concat([qa.sample(), tf.cast(qs.sample(), tf.float32)], axis=1)

        sex1, age1 = self.discriminate_0(z1)
        qs1 = tfd.Bernoulli(logits=sex1)
        qa1 = tfd.Multinomial(total_count=1., logits=age1)
        u_1 = tf.concat([qa.sample(), tf.cast(qs.sample(), tf.float32)], axis=1)

        def log_prob(v, a):
            v = tf.cast(v, tf.float32)
            return tf.reduce_mean(
                -tf.nn.softmax_cross_entropy_with_logits(logits=a, labels=v[:, :-1]) + qs.log_prob(v[:, -1:]))

        logits_y = self.classify(z1)
        qy = tfd.Bernoulli(logits=logits_y)
        return z1, u_, \
               qzx.log_prob(z1), \
               log_prob(u_, age), log_prob(u, age), \
               qy, log_prob(u, age0), log_prob(u, age1)


class VariationalDecoder(object):
    def __init__(self, z_dim=50, x_dim=71):
        self.z_dim = z_dim

        def decoder_func(z):
            fc1 = tf.layers.dense(z, 100, activation=tf.nn.softplus)
            logits = tf.layers.dense(fc1, x_dim, activation=tf.identity)
            return logits

        self.decoder = tf.make_template('decoder/x', lambda z: decoder_func(z))

    def sample_and_log_prob(self, z, x, u):
        z1 = z
        pz1 = tfd.MultivariateNormalDiag(loc=tf.zeros_like(z1), scale_diag=tf.ones_like(z1))
        x_ = self.decoder(tf.concat([z1, u], axis=1))  # p(x | z_1, u)
        pxz = tfd.Bernoulli(logits=x_)
        return x_, tf.reduce_sum(pxz.log_prob(x), axis=1), pz1.log_prob(z1)

    @property
    def vars(self):
        return [var for var in tf.global_variables() if 'decoder' in var.name]


@click.command()
@click.option('--mi', type=click.FLOAT, default=0.0)
@click.option('--e1', type=click.FLOAT, default=1.0)
@click.option('--e2', type=click.FLOAT, default=0.0)
@click.option('--e3', type=click.FLOAT, default=0.0)
@click.option('--e4', type=click.FLOAT, default=0.0)
@click.option('--e5', type=click.FLOAT, default=0.0)
@click.option('--disc', type=click.INT, default=10)
@click.option('--lag', is_flag=True, flag_value=True)
@click.option('--test', is_flag=True, flag_value=True)
@click.option('--gpu', type=click.INT, default=-1)
def main(mi, e1, e2, e3, e4, e5, disc, lag, test, gpu):
    test_bool = test
    import os
    if gpu == -1:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        device_id = tu.find_avaiable_gpu()
    else:
        device_id = gpu
    print('Using device {}'.format(device_id))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    try:
        jobid = os.environ['SLURM_JOB_ID']
    except:
        jobid = 0
    z_dim = 20
    train, test, pu = create_health_dataset(batch=64)
    datasets = tu.Datasets(train=train, test=test)
    # optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5)
    #
    global_step = tf.Variable(0, trainable=False, name='global_step')
    starter_learning_rate = 0.001
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               1000, 0.98, staircase=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5)
    encoder = VariationalEncoder(z_dim=z_dim)
    decoder = VariationalDecoder(z_dim=z_dim)
    if lag:
        logdir = tu.obtain_log_path('fair/lmifr_n/health/{}-{}-{}-{}-{}-{}-{}/'.format(mi, e1, e2, e3, e4, e5, disc))
    else:
        logdir = tu.obtain_log_path('fair/mifr_n/health/{}-{}-{}-{}-{}-{}-{}/'.format(mi, e1, e2, e3, e4, e5, disc))

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    with open(os.path.join(logdir, 'jobid'), 'w') as f:
        f.write(jobid)

    vae = LagrangianFairTransferableAutoEncoder(encoder, decoder, datasets, optimizer, logdir,
                                                pu, mi, e1, e2, e3, e4, e5, lag, disc)
    if not test_bool:
        vae.train(num_epochs=2000)
    vae.test_health()


if __name__ == '__main__':
    main()
