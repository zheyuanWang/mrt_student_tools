import cv2
import time
import os
from multiprocessing import cpu_count
from multiprocessing import Pool


def demo():
    start = time.time()

    def func(list):
        return 0
        # cv2.normalize(f, f, 0, 255, cv2.NORM_MINMAX)
        # cv2.imwrite(_path_f, f)


    pool = Pool(processes=6)
    pool.map(func, list)
    pool.close()
    pool.join()

    end = time.time()
    print("time: ", end - start)



if __name__ == '__main__':
    demo()
