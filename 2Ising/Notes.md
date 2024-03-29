# Notes on the Ising Model Simulation

## Be fast

I think that the greatest hurdle is the convolution. I am keeping notes from this file

https://laurentperrinet.github.io/sciblog/posts/2017-09-20-the-fastest-2d-convolution-in-the-world.html

| Method               | Time per loop, no prefetch [s] | with prefetch [s]|
| -------------------- | ------------------------------ | ---------------  |
| `scipy.convolve2D`      | 3.59  | 3.4   |
| `scipy.fft`             | 0.407 | 0.386 |
| `numpy.fft`             | 0.662 | 0.636 |
| `scipy.ndimage.convolve`| 1.48  | 1.5 |
| `skimage.feature.match_template`| 0.764  | 0.743 |
| `numba` | 1.2 | 1.44 |
| `pytorch` | 1.22 | 1.22 |



### Prefetching

From `scikit-learn`'s documentation

>Although the labeling of the axes seems arbitrary, it can have a significant effect on speed of operations. This is because modern processors never retrieve just one item from memory, but rather a whole chunk of adjacent items. (This is called prefetching.) Therefore, processing elements that are next to each other in memory is faster than processing them in a different order, even if the number of operations is the same:

>When the dimension you are iterating over is even larger, the speedup is even more dramatic. It is worth thinking about this data locality when writing algorithms. In particular, know that scikit-image uses C-contiguous arrays unless otherwise specified, so one should iterate along the last/rightmost dimension in the innermost loop of the computation.

They say that they see about **x8.6** speedup.