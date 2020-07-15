import numpy as np
import numpy
import h5py
import random
import math
from cupy.lib.stride_tricks import as_strided
import cupy
import cupy as cp
import time

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
            XF2 = cp.fft.fftshift(XF2, axes=-1)
            XF2 /= P

            i = (k+l) // 2
            a = int(((k-l)/float(Np) + 1.) * N)
            Sx[:,i,a-Mp:a+Mp] = XF2[:,(P//2-Mp):(P//2+Mp)]
    return Sx   # shape (batch, alpha, f_k)

# return alpha profile of the SCD matrix

def scf_per_batch(Np, L, xs):  # xs shape (bs,1024,2)
    B = Np//2
    s = scd_fam(xs, Np, L)  # shape (batch, alpha, f_k)
    f = cp.absolute(s)
    alpha = cp.amax(cp.absolute(s), axis=-1)
    freq = cp.amax(cp.absolute(s), axis=-2)
    (bs, my, mx) = f.shape 
    freq = freq[:, (mx//2-B):(mx//2 + B)]

    return alpha, freq   # should be (bs,Np) (bs,Np)


# transform train set
Np = 256
L = 1
bss = 1024
# path to the dataset.
x = h5py.File('../../dataset/201801a_data_test.h5', 'r+')
xx = x['test']
set_size = xx.shape[0]
cnt = int(set_size/bss)
data_test = []

with cupy.cuda.Device(1):
    for ii in range(cnt):
        s = xx[ii*bss:(ii+1)*bss,:,0] + 1j * xx[ii*bss:(ii+1)*bss,:,1]
        s = cp.array(s)
    
        alpha, freq = scf_per_batch(Np, L, s)  #(bs,Np) (bs,Np)
        # should be (bs,Np,2)
        x_gpu = cp.concatenate((alpha, freq), axis=1).reshape((bss,2,Np))
        x_cpu = cp.asnumpy(x_gpu)
        x_ = np.transpose(x_cpu,(0,2,1))
        print (x_.shape, ii)
        data_test.append(x_)

data = np.asarray(data_test).reshape(set_size,Np,2)
hf = h5py.File('201801a_scf_256_test.h5', 'w')
hf.create_dataset('test_scf', data=data)
hf.close()
