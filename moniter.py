import os, sys


def latest_file(dir_input):
    lst = os.listdir(dir_input)
    lst.sort(key=lambda fn: os.path.getmtime(dir_input+'/'+fn))
    for f in os.listdir(dir_input):
        if f != lst[-1]:
            os.remove(os.path.join(dir_input, f))


if __name__ == '__main__':
    tb_basic_dir = './log/tb/{}'.format(sys.argv[1])
    tb_file = latest_file(tb_basic_dir)
    os.system("tensorboard --logdir={} --bind_all".format(tb_basic_dir))




