import multiprocessing
import multiprocessing.connection

# def worker_process(remote: multiprocessing.connection.Connection, seed: int):
#     game = Game(seed)
#     while True:
#         cmd, data = remote.recv()
#         if cmd == "step":
#             remote.send(game.step(data)) # send back results from a step
#         elif cmd == "reset":
#             remote.send(game.reset())
#         elif cmd == "close":
#             remote.close()
#             break
#         else:
#             raise NotImplementedError


def worker_process(remote: multiprocessing.connection.Connection, seed: int):
    while True:
        cmd, data = remote.recv()
        if cmd == 'reset':
            remote.send('Resetted!')
        elif cmd == 'close':
            remote.close()
            break
        else:
            raise NotImplementedError


class Worker:
    child: multiprocessing.connection.Connection
    process: multiprocessing.Process

    def __init__(self, seed):
        self.child, parent = multiprocessing.Pipe()
        self.process = multiprocessing.Process(
            target=worker_process, args=(parent, seed))
        self.process.start()


class Main():
    def __init__(self):
        self.workers = [Worker(42) for _ in range(8)]
        self.obs = list()
        for worker in self.workers:
            worker.child.send(("reset", None))
            print('sent')
        for i, worker in enumerate(self.workers):
            self.obs.append(worker.child.recv())

    def p(self):
        print(self.obs)


if __name__ == "__main__":
    m = Main()
    m.p()
