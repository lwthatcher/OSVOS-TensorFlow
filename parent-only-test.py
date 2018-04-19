import os
import argparse
import tensorflow as tf
import osvos
from dataset import Dataset


def _sanitize_kwargs(kwargs):
    keys_to_remove = []
    for k,v in kwargs.items():
        if v is None:
            keys_to_remove.append(k)
    for k in keys_to_remove:
        del kwargs[k]
    return kwargs


# Test the network
def test_parent(seq_name, **kwargs):
    # result path
    result_path = os.path.join('DAVIS', 'Results', 'Segmentations', '480p', 'OSVOS-parent', seq_name)
    # Define Dataset
    test_frames = sorted(os.listdir(os.path.join('DAVIS', 'JPEGImages', '480p', seq_name)))
    test_imgs = [os.path.join('DAVIS', 'JPEGImages', '480p', seq_name, frame) for frame in test_frames]
    dataset = Dataset(None, test_imgs, './')

    with tf.Graph().as_default():
        checkpoint_path = os.path.join('models', 'OSVOS_parent', 'OSVOS_parent.ckpt-50000')
        osvos.test(dataset, checkpoint_path, result_path)


if __name__ == "__main__":
    # define acceptable sequences
    _path = os.path.join('DAVIS', 'JPEGImages', '480p')
    known_sequences = next(os.walk(_path))[1]
    # define parser
    parser = argparse.ArgumentParser('parameters for running OSVOS')
    parser.add_argument('seq_name', choices=known_sequences, help='the name of the sequence to run')
    args = parser.parse_args()
    # sequence name
    seq = args.seq_name
    # kwargs
    kwargs = vars(args)
    del kwargs['seq_name']
    kwargs = _sanitize_kwargs(kwargs)
    print('Parameters:', kwargs)
    # run learner
    test_parent(seq, **kwargs)
