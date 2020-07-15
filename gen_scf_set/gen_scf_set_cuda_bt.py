import numpy as np
import numpy
import h5py
import random
import math
from cupy.lib.stride_tricks import as_strided
import cupy
import cupy as cp


def scd_fam(x, Np, L, N=None):   #  x = 1024 Np = 256 L = 1
    def sliding_window(x, w, s):
        shape = (x.shape[0], ((x.shape[1] - w) // s + 1), w)
        strides = (x.strides[0], x.strides[1]*s, x.strides[1])
        return as_strided(x, shape, strides)

    # input channelization
    bs = x.shape[0]
    xs = sliding_window(x, Np, L)
    
    Pe = int(cp.floor(int(cp.log(xs.shape[1])/cp.log(2))))
    P = 2**Pe
    N = L*P

    xs2 = xs[:,0:P,:]
    # windowing
    w = cp.hamming(Np)
    w /= cp.sqrt(cp.sum(w**2))
    xw = xs2 * cp.tile(w, (P,1))
    XF1 = cp.fft.fft(xw, axis=-1)
    XF1 = cp.fft.fftshift(XF1, axes=-1)
    # calculating complex demodulates
    f = cp.arange(Np)/float(Np) - .5
    t = cp.arange(P)*L

    f = cp.tile(f, (P,1))
    t = cp.tile(t.reshape(P,1), (1, Np))

    XD = XF1
    XD *= cp.exp(-1j*2*np.pi*f*t)

    # calculating conjugate products, second FFT and the final matrix
    Sx = cp.zeros((bs, Np, 2*N), dtype=cp.complex64)
    Mp = N//Np//2

    for k in range(Np):
        for l in range(Np):
            XF2 = cp.fft.fft(XD[:,:,k]*cp.conj(XD[:,:,l]), axis=-1)
            XF2 = cp.fft.fftshift(XF2, axis=-1)
            XF2 /= P

            i = (k+l) // 2
            a = int(((k-l)/float(Np) + 1.) * N)
            Sx[:,i,a-Mp:a+Mp] = XF2[:,(P//2-Mp):(P//2+Mp)]
    return Sx   # shape (alpha, f_k)

# return alpha profile of the SCD matrix
def alphaprofile(s):
    return cp.amax(cp.absolute(s), axis=-1)

def freqprofile(s):
    return cp.amax(cp.absolute(s), axis=-2)    

def scf_per_signal(Np, L, xs):  # xs shape (1024,2)
    B = Np//2
    s = scd_fam(xs, Np, L)
    f = cp.absolute(s)
    alpha = alphaprofile(s)
    freq = freqprofile(s)
    (bs, my, mx) = f.shape 
    freq = freq[:, (mx//2-B):(mx//2 + B)]
    print ('done')
    return alpha, freq   # should be (256,2)


# transform train set
Np = 256
L = 1
# path to the dataset.
x = h5py.File('../../dataset/201801a_data_train.h5', 'r+')
xx = x['train']
data_train = []

s = xx[:256,:,0] + 1j * xx[:256,:,1]
s = cp.array(s)
oo, o = scf_per_signal(Np, L, s)
'''
data_train.append(oo)
print (oo.shape)

hf = h5py.File('201801a_scf_256.h5', 'w')
hf.create_dataset('train_scf', data=data_train)
hf.close()
'''
