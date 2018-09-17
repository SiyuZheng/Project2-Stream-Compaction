CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Siyu Zheng
* Tested on: Windows 10, i7-8750 @ 2.20GHz 16GB, GTX 1060 6GB (Personal Laptop)

```

****************
** SCAN TESTS **
****************
    [  10  39  41   0  14  37  18  40   1  42  27  21  10 ...  14   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 0.10945ms    (std::chrono Measured)
    [   0  10  49  90  90 104 141 159 199 200 242 269 290 ... 803563 803577 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 0.116406ms    (std::chrono Measured)
    [   0  10  49  90  90 104 141 159 199 200 242 269 290 ... 803493 803514 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 0.235072ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 0.197024ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 0.147424ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 0.119808ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 0.299008ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 0.253952ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   0   3   0   2   1   3   2   3   3   1   3   0   0 ...   1   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 0.121971ms    (std::chrono Measured)
    [   3   2   1   3   2   3   3   1   3   2   1   2   1 ...   2   1 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 0.139594ms    (std::chrono Measured)
    [   3   2   1   3   2   3   3   1   3   2   1   2   1 ...   1   1 ]
    passed
==== cpu compact with scan ====
   elapsed time: 0.552812ms    (std::chrono Measured)
    [   3   2   1   3   2   3   3   1   3   2   1   2   1 ...   2   1 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 0.141568ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 0.243712ms    (CUDA Measured)
    passed

```
