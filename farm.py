# farm.py
# a single instance of a farm.

# a farm should consist of a pool of instances
# and expose those instances as one giant callable class

import multiprocessing, time, random, threading
from multiprocessing import Process, Pipe, Queue

import traceback

from create_env import create_env

# for hacking seeds
from test_multi import set_all_seeds
#import numpy as np



ncpu = multiprocessing.cpu_count()



# separate process that holds a separate RunEnv instance.
# This has to be done since RunEnv() in the same process result in interleaved running of simulations.
def standalone_headless_isolated(pq, cq, plock):
    # locking to prevent mixed-up printing.
    plock.acquire()
    print('starting headless...',pq,cq)
    try:
        # create a training environment
        env = create_env(train=True, render=False)
    except Exception as ex:
        print('error on start of standalone')
        traceback.print_exc()
        plock.release()
        return
    else:
        plock.release()

    def report(env):
        # a way to report errors ( since you can't just throw them over a pipe )
        # env should be a string
        print('(standalone) got error!!!')
        cq.put(('error',env))

    #def floatify(np):
    #    return [float(np[i]) for i in range(len(np))]

    try:
        while True:
            msg = pq.get()
            # messages should be tuples,
            # msg[0] should be string

            if msg[0] == 'reset':

                # only play bad episodes
                #seeds = [4, 7, 8, 10, 15, 20, 24, 25, 26, 30, 31, 39, 44, 45, 50, 56, 57, 58, 59]
                #seed = random.choice(seeds)
                #set_all_seeds(seed)
                #print("Picked seed", seed)

                o = env.reset(project=False)
                #o = env.reset()                                # for gym env
                #o = floatify(o)                                # obs is np in gym env
                # send as tuple
                cq.put((o,))
            elif msg[0] == 'step':
                o,r,d,i = env.step(msg[1], project=False)
                #o,r,d,i = env.step(msg[1])                     # for gym env
                #o = floatify(o)
                cq.put((o,r,d,i))
            else:
                cq.close()
                pq.close()
                del env
                break
    except Exception as ex:
        traceback.print_exc()
        report(str(ex))

    return # end process

# global process lock
plock = multiprocessing.Lock()
# global thread lock
tlock = threading.Lock()

# global id issurance
eid = int(random.random()*100000)
def get_eid():
    global eid,tlock
    tlock.acquire()
    i = eid
    eid+=1
    tlock.release()
    return i

# class that manages the interprocess communication and expose itself as a RunEnv.
# reinforced: this class should be long-running. it should reload the process on errors.

class ei: # Environment Instance
    def __init__(self):
        self.occupied = False # is this instance occupied by a remote client
        self.id = get_eid() # what is the id of this environment
        self.pretty('instance creating')

        self.newproc()
        import threading as th
        self.lock = th.Lock()

    def timer_update(self):
        self.last_interaction = time.time()

    def is_occupied(self):
        if self.occupied == False:
            return False
        else:
            if time.time() - self.last_interaction > 20*60:
                # if no interaction for more than X minutes
                self.pretty('no interaction for too long, self-releasing now. applying for a new id.')

                self.id = get_eid() # apply for a new id.
                self.occupied == False

                self.pretty('self-released.')

                return False
            else:
                return True

    def occupy(self):
        self.lock.acquire()
        if self.is_occupied() == False:
            self.occupied = True
            self.id = get_eid()
            self.lock.release()
            return True # on success
        else:
            self.lock.release()
            return False # failed

    def release(self):
        self.lock.acquire()
        self.occupied = False
        self.id = get_eid()
        self.lock.release()

    # create a new RunEnv in a new process.
    def newproc(self):
        global plock
        self.timer_update()

        self.pq, self.cq = Queue(1), Queue(1) # two queue needed

        self.p = Process(
            target = standalone_headless_isolated,
            args=(self.pq, self.cq, plock)
        )
        self.p.daemon = True
        self.p.start()

        self.reset_count = 0 # how many times has this instance been reset
        self.step_count = 0

        self.timer_update()
        return

    # send x to the process
    def send(self,x):
        return self.pq.put(x)

    # receive from the process.
    def recv(self):
        # receive and detect if we got any errors
        r = self.cq.get()

        #print(r)

        # isinstance is dangerous, commented out
        # if isinstance(r,tuple):
        if r[0] == 'error':
            # read the exception string
            e = r[1]
            self.pretty('got exception')
            self.pretty(e)
            raise Exception(e)
        return r

    def reset(self, project=False):
        self.timer_update()
        if not self.is_alive():
            # if our process is dead for some reason
            self.pretty('process found dead on reset(). reloading.')
            self.kill()
            self.newproc()

        if self.reset_count>50 or self.step_count>10000: # if reset for more than X times
            self.pretty('Environment has been reset too many times. Reloading...')
            self.kill()
            self.newproc()

        self.reset_count += 1
        self.send(('reset',))
        r = self.recv()
        self.timer_update()
        # return first of tuple
        return r[0]

    def step(self, actions, project=False):
        self.timer_update()
        self.send(('step',actions,))
        r = self.recv()
        self.timer_update()
        self.step_count+=1
        return r

    def kill(self):
        if not self.is_alive():
            self.pretty('process already dead, no need for kill.')
        else:
            self.send(('exit',))
            self.pretty('waiting for join()...')

            while 1:
                self.p.join(timeout=5)
                if not self.is_alive():
                    break
                else:
                    self.pretty('process is not joining after 5s, still waiting...')

            self.pretty('process joined.')

    def __del__(self):
        self.pretty('__del__')
        self.kill()
        self.pretty('__del__ accomplished.')

    def is_alive(self):
        return self.p.is_alive()

    # pretty printing
    def pretty(self,s):
        print(('(ei) {} ').format(self.id)+str(s))

# class that other classes acquires and releases EIs from.
class eipool: # Environment Instance Pool
    def pretty(self,s):
        print(('(eipool) ')+str(s))

    def __init__(self,n=1):
        import threading as th
        self.pretty('starting '+str(n)+' instance(s)...')
        self.pool = [ei() for i in range(n)]
        self.lock = th.Lock()

    def acq_env(self):
        self.lock.acquire()
        for env in self.pool:
            if env.occupy() == True: # successfully occupied an environment
                self.lock.release()
                return env # return the envinstance
        self.lock.release()
        return False # no available ei

    def rel_env(self,ei):
        self.lock.acquire()
        for env in self.pool:
            if env == ei:
                env.release() # freed
        self.lock.release()

    def get_env_by_id(self,id):
        for e in self.pool:
            if e.id == id:
                return e
        return False

    def __del__(self):
        for e in self.pool:
            del e

# farm
# interface with eipool via eids.
# ! this class is a singleton. must be made thread-safe.
import traceback
class farm:
    def pretty(self,s):
        print(('(farm) ')+str(s))

    def __init__(self):
        # on init, create a pool
        # self.renew()
        import threading as th
        self.lock = th.Lock()

    def acq(self,n=None):
        self.renew_if_needed(n)
        result = self.eip.acq_env() # thread-safe
        if result == False:
            ret = False
        else:
            self.pretty('new env '+str(result.id))
            ret = result.id
        return ret

    def rel(self,id):
        e = self.eip.get_env_by_id(id)
        #if e == False:
        #    self.pretty(str(id)+' not found on rel(), might already be released')
        #else:
        if e != False:
            self.eip.rel_env(e)
            #self.pretty('rel '+str(id))

    def step(self,id,actions):
        e = self.eip.get_env_by_id(id)
        if e == False:
            self.pretty(str(id)+' not found on step(), might already be released')
            return False

        try:
            ordi = e.step(actions)
            return ordi
        except Exception as e:
            traceback.print_exc()
            raise e

    def reset(self,id):
        e = self.eip.get_env_by_id(id)
        if e == False:
            self.pretty(str(id)+' not found on reset(), might already be released')
            return False

        try:
            oo = e.reset()
            return oo
        except Exception as e:
            traceback.print_exc()
            raise e

    def renew_if_needed(self,n=None):
        self.lock.acquire()
        if not hasattr(self,'eip'):
            self.pretty('renew because no eipool present')
            self._new(n)
        self.lock.release()

    def forcerenew(self,n=None):
        self.lock.acquire()
        self.pretty('forced pool renew')

        if hasattr(self,'eip'): # if eip exists
            del self.eip
        self._new(n)
        self.lock.release()

    def _new(self,n=None):
        self.eip = eipool(ncpu if n is None else n)

# expose the farm via Pyro4
def main():
    from pyro_helper import pyro_expose
    pyro_expose(farm,20099,'farm')

if __name__ == '__main__':
    main()
