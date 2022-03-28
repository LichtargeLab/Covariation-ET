/* Chen Wang @ 03/23/2022 */
// [[Rcpp::plugins(cpp11)]]
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
/* This function calculates the trace for a given group
 * Matrix x is the non.concert.mat. A non concerted element in the 
 * matrix is represented as a 8 digit integer.
 */
double TraceGroupC(int start, int end, int group_count, IntegerMatrix x) {
  double out;
  std::vector<int> vec;
  /* This block index the non concerted elements inside the group*/
  vec.reserve(group_count);
  for (int i = start-1; i < end; i++) {
    for (int j = i; j < end; j++) {
      if (x(i,j) !=0 ){
        vec.push_back(x(i,j));
      }
    }
  }

  if (vec.size() == 0) {
    /* If no non concerted variant is present
     * this group has a trace of 1.
     */
    out = 1;
  } else {
    /* This block counts the occurrence for each non 
     * concerted type that present in the group.
     */
    std::map<int, double> counts;
    int n = vec.size();
    for (int i = 0; i < n; i++) {
      counts[vec[i]]++;
      }
    /* This block calculates the CovET trace score for
     * this group.
     */
    out = 0;
    double freq = 0;
    for(auto elem : counts) {
      freq = elem.second/group_count;
      out += freq*log(freq);
      }
    out = exp(-out);
  }
  return out;
}


// [[Rcpp::export]]
NumericVector TraceGroupC_vec(IntegerVector start, IntegerVector end, IntegerVector group_count, IntegerMatrix x) {
  int n = start.size();
  NumericVector out(n);
  for (int i = 0; i < n; ++i) {
    out[i] = TraceGroupC(start[i], end[i], group_count[i], x);
    }
  return out;
}
