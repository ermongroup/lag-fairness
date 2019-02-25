import subprocess
import shlex
import os
import datetime
import shutil
from itertools import product
import argparse


def execute(cmd, jobname, ngpus=1, partition='atlas', debug=False, max_time=2880):
    if debug:
        print(cmd)
    else:
        slurm_cmd = 'sbatch --output=/atlas/u/tsong/stdout/%j.out --error=/atlas/u/tsong/stderr/%j.err' \
                     + ' --nodes=1 --ntasks-per-node=1 --time={}'.format(max_time) \
                     + ' --mem={}G --partition={} --cpus-per-task={}'.format(ngpus*16, partition, ngpus*4+2) \
                     + ' --gres=gpu:{} --job-name={} --wrap=\"{}\"'.format(ngpus, jobname, cmd)
        print(slurm_cmd)
        process = subprocess.Popen(slurm_cmd, shell=True)


def create_command(script, options, switches, keys):
    cmd = 'python {} '.format(script)

    for k, v in options.items():
        ks = k.replace('_', '-')
        vs = str(v)
        cmd += ' --{} {}'.format(ks, vs)

    for s in switches:
        cmd += ' {}'.format(s)

    jobname = '-'.join([k for k in keys])
    return cmd, jobname


def copy_and_teleport(src, dst):
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    dst = os.path.join(dst, now)
    shutil.copytree(src, dst)
    os.chdir(dst)
    print('Moved to directory {}'.format(dst))


class Task(object):
    def __init__(self):
        self.options = dict()
        self.switches = list()
        self.keys = list()


def launch(args, script, kwargs, debug=True, partition='atlas', ngpus=1, max_time=4320):
    task = Task()

    task.options.update({
        'mi': kwargs.get('mi', 0.0),
        'e1': kwargs.get('e1', 0.0),
        'e2': kwargs.get('e2', 0.0),
        'e3': kwargs.get('e3', 1.0),
        'e4': kwargs.get('e4', 0.0),
        'e5': kwargs.get('e5', 0.0),
        'disc': kwargs.get('disc', 1),
    })

    if args.lag == True:
        lag_str = 'lag'
    else:
        lag_str = ''

    task.keys.extend([
        args.task, lag_str
    ])

    cmd, jobname = create_command(script, task.options, task.switches, task.keys)
    execute(cmd, jobname, debug=debug, partition=partition, ngpus=ngpus, max_time=max_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--partition', type=str, default='atlas')
    parser.add_argument('--task', type=str, default='adult')
    parser.add_argument('--time', type=int, default=4320)
    parser.add_argument('--lag', action='store_true')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    if not args.debug:
        s = input('Trying to submit jobs! Enter y to continue.')
        if s[0] != 'y':
            print('Jobs not submitted!')
            exit(0)

    if not args.debug:
        src = os.getcwd()
        dst = '/atlas/u/tsong/dump/fair'
        copy_and_teleport(src, dst)

    ngpus = 1
    from itertools import product

    if args.lag:
        if args.task == 'adult':
            e1s = [10]
            e2s = [0.10]#[0.02, 0.05, 0.10, 0.15]
            e3s = [16] #[13, 14, 15, 16]
        elif args.task == 'health':
            e1s = [15]#[5, 10, 15, 20]
            e2s = [0.3]#[0.1, 0.2, 0.3, 0.4]
            e3s = [23]#[22, 23, 24, 25]
        elif args.task == 'german':
            e1s = [5, 12, 19, 26]
            e2s = [0.1, 0.2, 0.3, 0.4]
            e3s = [18, 22, 26, 30]
    else:
        e1s = [0.1, 0.2, 1.0, 2.0]
        e2s = [0.1, 0.2, 1.0, 2.0, 5.0]
        e3s = [0.0, 0.1, 0.2, 1.0, 2.0]
    # [0.1, 0.2, 1.0, 2.0], # [2, 6, 10, 14]
    # [0.1, 0.2, 1.0, 2.0], # [0.02, 0.05, 0.10, 0.15]
    # [0.1, 0.2, 1.0, 2.0], # [13, 14, 15, 16]

    for mi, e1, e2, e3, disc in product(
        [1.0],
        e1s,
        e2s,
        e3s,
        [10]
    ):
        script = '-m examples.{}'.format(args.task)
        if args.lag:
            script = script + ' --lag '
        if args.test:
            script = script + ' --test '
        kwargs = {
            'mi': mi,
            'e1': e1 / mi,
            'e2': e2 / mi,
            'e3': e3 / mi,
            'e4': 0.0,
            'e5': 0.0,
            'disc': disc
        }
        launch(args, script=script, kwargs=kwargs, max_time=args.time, debug=args.debug, ngpus=ngpus)
