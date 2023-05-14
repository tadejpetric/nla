//
// Created by user on 4/27/23.
//

#ifndef NLA_NAL1_H
#define NLA_NAL1_H
#include <cmath>
#include <chrono>
#include <iostream>

struct area {
    const int divisions;
    const int k;

    // step_width2 is square of step width
    // h = 2/(divisions+1). -1 because (-1,1) not [-1,1)
    // eg if we have 3 divisions, -> {-0.5, 0, 0.5}
    //              i = 0 -> -0.5,    i = 2 -> 0.5
    const double step_width2;

    area(const int divisions, const int k) : divisions(divisions),
                                             k(k),
                                             step_width2(4./((divisions+1)*(divisions+1))) {}

    double itoc(const int i) const {
        return (i+1)*2./(divisions+1) -1;
    }

    int ctoi(const double x) const {
        return std::lround((x+1)*(divisions+1)/2. - 1);
    }

    double penalty(int i, int j) const {
        double x = itoc(i);
        double y = itoc(j);
        if (x < -1 || x > 1) return k;
        if (y < -1 || y > 1) return k;
        if (x*x + y*y < 1/10.) return k;

        return 0;
    }

};

std::chrono::time_point<std::chrono::high_resolution_clock> start() {
    return std::chrono::high_resolution_clock::now();
}

void stop(std::chrono::time_point<std::chrono::high_resolution_clock> t) {
    using std::chrono::high_resolution_clock;
    using std::chrono::duration;
    duration<double> delta = high_resolution_clock::now() - t;
    std::cout << "duration: " << delta.count() << "s\n";
}



#endif //NLA_NAL1_H
