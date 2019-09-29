import multiprocessing
import time

start = time.perf_counter()


def do_something(seconds):
    print(f'Sleeping {seconds} second...')
    time.sleep(seconds)
    print('Done sleeping...')


def main():
    processes = []

    for _ in range(10):
        p = multiprocessing.Process(target=do_something, args=[1.5])
        p.start()
        processes.append(p)

    finish = time.perf_counter()

    print(f'Finished in {round(finish-start,2)} second(s)')


if __name__ == '__main__':
    main()


'''
    The arguments to a process have to be able to be serialized by Pickle. this means we are converting python objects that 
    can be deconstructed and reconstructed in another script. 

'''
