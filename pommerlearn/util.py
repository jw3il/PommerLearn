import concurrent.futures
import time


def duration_of(fun):
    start = time.time()

    fun()

    duration = time.time() - start
    print("Duration: {}".format(duration))
    return duration


def pooled_multi_run(fun, args, count, status_fun=(lambda res: print('Jobs completed: {}'.format(len(res)))), wait=0.5):
    """
    Executes a given function multiple times with the same parameters using a ProcessPoolExecutor.

    :param fun: The function which shall be executed
    :param args: The arguments of the function
    :param count: How often fun(*args) should be executed
    :param status_fun: A custom function which receives the current results as an input
    :param wait: How long to wait between checking for new results
    :return: The list of results containing the individual function executions
    """
    results = []

    futures = set()
    futures_to_remove = set()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        # submit all jobs
        for a in range(0, count):
            future = executor.submit(fun, *args)
            futures.add(future)

        # wait until they are done
        while True:
            change = False

            for f in futures:
                if f.done():
                    return_value = f.result()
                    results.append(return_value)
                    change = True
                    futures_to_remove.add(f)

            if change:
                status_fun(results)
                futures = futures - futures_to_remove
                futures_to_remove = set()

            # wait until all executors are done
            if len(futures) == 0:
                break

            # wait some time before checking again
            time.sleep(wait)

        return results