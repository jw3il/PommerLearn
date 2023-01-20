#include "aggregate.h"
#include <limits>

void OnlineAggregate::reset()
{
    mean = 0;
    count = 0;
    M2 = 0;
    max = -std::numeric_limits<double>::infinity();
    min = std::numeric_limits<double>::infinity();
}

void OnlineAggregate::update(double value)
{
    count += 1;
    double delta = value - mean;
    mean += delta / count;
    double delta2 = value - mean;
    M2 += delta * delta2;

    max = value > max ? value : max;
    min = value < min ? value : min;
}

double OnlineAggregate::get_count()
{
    return count;
}

double OnlineAggregate::get_mean()
{
    return mean;
}

double OnlineAggregate::get_max()
{
    return max;
}

double OnlineAggregate::get_min()
{
    return min;
}

double OnlineAggregate::get_variance()
{
    return M2 / count;
}

double OnlineAggregate::get_sample_variance()
{
    return M2 / (count - 1);
}

std::string OnlineAggregate::get_csv()
{
    return std::to_string(get_count()) + "," + std::to_string(get_mean())\
         + "," + std::to_string(get_max()) + "," + std::to_string(get_min())\
         + "," + std::to_string(get_variance()) + "," + std::to_string(get_sample_variance());
}