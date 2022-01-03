#ifndef SAFE_QUEUE
#define SAFE_QUEUE

#include <queue>
#include <mutex>
#include <condition_variable>

/**
 * @brief Thread-safe queue of unique pointers.
 * 
 * Based on https://stackoverflow.com/questions/15278343/c11-thread-safe-queue.
 *  
 * @tparam T element type
 */
template <class T>
class SafePtrQueue
{
public:
  SafePtrQueue(void)
    : q()
    , m()
    , c()
  {}

  ~SafePtrQueue(void)
  {}

  /**
   * @brief Adds an element to the queue.
   * 
   * @param t pointer of element to be added
   */
  void enqueue(std::unique_ptr<T> t)
  {
    std::lock_guard<std::mutex> lock(m);
    q.push(std::move(t));
    c.notify_one();
  }

  /**
   * @brief Remove the front element from the queue and return it.
   * Waits until elements are available if the queue is empty.
   * 
   * @return T pointer to old front element
   */
  std::unique_ptr<T> dequeue(void)
  {
    std::unique_lock<std::mutex> lock(m);
    while(q.empty())
    {
      // release lock as long as the wait and reaquire it afterwards.
      c.wait(lock);
    }
    std::unique_ptr<T> val = std::move(q.front());
    q.pop();
    return val;
  }

private:
  std::queue<std::unique_ptr<T>> q;
  mutable std::mutex m;
  std::condition_variable c;
};
#endif
