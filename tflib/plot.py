import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import collections
import time
#import cPickle as pickle
import pickle

_since_beginning = collections.defaultdict(lambda: {})
_since_last_flush = collections.defaultdict(lambda: {})

_iter = [0]
def tick():
        _iter[0] += 1

def plot(name, value):
	_since_last_flush[name][_iter[0]] = value

def flush(outdir=None):
	prints = []
	for name, vals in list(_since_last_flush.items()):
		#prints.append("{}\t{}".format(name, np.mean(list(vals.values()))))
		prints.append("{}\t{:.8g}".format(name[-5:], np.mean(list(vals.values()))))
		_since_beginning[name].update(vals)

		x_vals = np.sort(list(_since_beginning[name].keys()))
		y_vals = [_since_beginning[name][x] for x in x_vals]

		fig = plt.gcf()
		sizes = [4000,3000]
		dpi = 300
		fig.set_size_inches(sizes[0] / dpi, sizes[1] / dpi, forward = False)
		plt.clf()
		plt.plot(x_vals, y_vals)
		plt.xlabel('iteration')
		plt.ylabel(name[-5:])

		if outdir:
        		plt.savefig(outdir + '/' + name.replace(' ', '_')+'.png')
		else:
			plt.savefig(name.replace(' ', '_')+'.png')

	print(("iter {}\t{}".format(_iter[0], "\t".join(prints))))

	_since_last_flush.clear()

def dump(outdir):
        with open(outdir + '/log.pkl', 'wb') as f:
                pickle.dump(dict(_since_beginning), f, pickle.HIGHEST_PROTOCOL)
