//
// Created by user on 4/27/23.
//

#ifndef NLA_NAL1_H
#define NLA_NAL1_H
#include <cmath>
struct area {
    const int divisions;
    area(const int divisions) : divisions(divisions) {}

    double itoc(const int i) {
        return i*2./(divisions-1) -1;
    }

    int ctoi(const double x) {
        return std::lround((x+1)*(divisions-1)/2.);
    }

    double penalty(int i, int j, double k) {
        double x = itoc(i);
        double y = itoc(j);
        if (x < -1 || x > 1) return k;
        if (y < -1 || y > 1) return k;
        if (x*x + y*y < 1/10.) return k;

        return 0;
    }
};


#endif //NLA_NAL1_H
