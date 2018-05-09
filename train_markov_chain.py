import numpy as np
from sparse import COO, DOK
import sys
import pickle
import argparse
from markov_model import update_progress

parser = argparse.ArgumentParser(description='markov_model.py')

parser.add_argument('-save_model', default='markov.pkl',
                    help="""Model filename to save""")
parser.add_argument('-data', default='./data/input.txt',
                    help="""Filename for source text""")
parser.add_argument('-order', type=int, default=1,
                    help="""order of markov model.""")
parser.add_argument('-num_bytes', type=int, default=100000,
                    help="""Specify the number of bytes to train on.""")
parser.add_argument('-byte_class', default='./data/unique_bytes.npy',
                    help="""location of class list""")


opt = parser.parse_args()
unique_bytes = np.load(opt.byte_class)
byte_classes = dict()
#Encode unique bytes
for i, byt in enumerate(unique_bytes):
    #correct off-by-one issue
    byt = int(byt)
    byt -= 1
    byt = str(byt)
    byte_classes[byt] = i

#Add 'other' class in case unexpected byte encountered.
byte_classes['other'] = i + 1

def markovize_bytes(byte_string, order=1, class_dict=byte_classes):
    """Takes a byte string and iterates through it,
    taking a count of next bytes depending on previous state.
    Second order."""
    dims = tuple([len(class_dict) for i in range(order+1)])
    mat = DOK(shape=dims)
    prev = [-1 for i in range(order)]
    for i, true_byte in enumerate(byte_string):
        #reference dict to get byte index
        try:
            byt = class_dict[str(true_byte)]
        except KeyError:
            byt = class_dict['other']
        if not any(np.array(prev) < 0):
            loc = tuple(prev + [byt])
            mat[loc] += 1
        for j, val in enumerate(prev):
            if j == (len(prev) - 1):
                prev[j] = byt
            else:
                prev[j] = prev[j+1]
        #prog = (i+1)/len(byte_string)
        #update_progress(prog)
    return mat

def normalize_vecs(mat):
    """Normalizes probability vectors so they
    sum to one."""
    #convert to array for operation
    order = len(mat.shape) - 1
    mat = COO(mat)
    row_sums = mat.sum(axis=order)
    mat = DOK(mat)
    for point in mat.data:
        divisor = row_sums[point[:-1]]
        mat[point] = mat[point] / divisor
    mat = COO(mat)
    return mat

def train_markov_model(byte_string, order=1):
    """First or second order markov. Returns a probability matrix
    given a string of bytes."""
    print('generating probability matrix...')
    mat = markovize_bytes(byte_string, order=order)
    print('normalizing probabilities...')
    normed_mat = normalize_vecs(mat)
    return normed_mat

if __name__ in '__main__':
    text_source = opt.data
    print('loading training text from: ', text_source)
    with open(text_source, 'rb') as f:
        txt = f.read()

    order = opt.order
    print('training markov model of order: ', order)
    model = train_markov_model(txt[:opt.num_bytes], order=order)


    save_path = opt.save_model
    print('saving model to: ', save_path)
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
    
