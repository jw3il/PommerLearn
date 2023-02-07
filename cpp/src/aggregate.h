#ifndef AGGREGATE_H
#define AGGREGATE_H
#include <string>

/**
 * @brief Computes aggregate statistics using Welford's online algorithm.
 * 
 * See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
 */
class OnlineAggregate {
private:
    double count;
    double mean;
    double M2;
    double max;
    double min;
public:
    OnlineAggregate() 
    {
        reset();
    }
    
    /**
     * @brief Reset the running stats.
     */
    void reset();

    /**
     * @brief Updates the running stats with a new value.
     * 
     * @param value the new value
     */
    void update(double value);

    double get_count();
    double get_mean();
    double get_max();
    double get_min();
    double get_variance();
    double get_sample_variance();
    std::string get_csv();
};

#endif // AGGREGATE_H