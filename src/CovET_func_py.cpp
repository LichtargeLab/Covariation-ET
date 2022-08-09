//
// Created by spencer on 7/26/22.
//
#include <iostream>
#include <bits/stdc++.h>

double TraceGroupC(int start, int end, int group_count, int **x) {
    //std::cout << "start: " << start << " end: " << end << " group_count: " << group_count << "\n";
    double out;
    std::vector<int> vec;
    /* This block fills a vector with the non-concerted variations*/
    vec.reserve(group_count);
    for (int i = start; i <= end; i++) {
        for (int j = i; j <= end; j++) {
            //std::cout << "x[i][j] = " << x[i][j] << "\n";
            if (x[i][j] !=0 ){
                vec.push_back(x[i][j]);
            }
        }
    }
    if (vec.empty())
        //std::cout << "Vec.size = 0 \n";
        /* If no non concerted variant is present
         * this group has a trace of 1.
         */
        out = 1;
    else {
        /* This block counts the occurrence for each non
         * concerted type that present in the group.
         */
        std::map<int, double> counts;
        unsigned long n;
        n = vec.size();
        for (int i = 0; i < n; i++) {
            counts[vec[i]]++;
        }
        /* This block calculates the CovET trace score for
         * this group.
         */
        out = 0;
        double freq;
        for (auto elem: counts) {
            freq = elem.second / group_count;
            out += freq * std::log(freq);
        }
        out = exp(-out);
    }
    return out;
}
extern "C" double * TraceGroupC_vec(int *start, int *end, int *group_count,
                                 int n, int **non_concerted_matrix) {
    auto *out = new double[n];
    for (int i = 0; i < n; i++) {
        out[i] = TraceGroupC(start[i], end[i], group_count[i], non_concerted_matrix);
    }
    return out;
}

extern "C" void free_mem(const double* a)
{
delete[] a;
}

// g++ -c -fPIC CovET_func_py.cpp -o CovET_func_py.o
// g++ -shared -Wl,-soname,CovET_func_py.so -o CovET_func_py.so CovET_func_py.o