extern "C" long long mysum(int n, int* array) {
    long long res = 0;
    for (int i = 0; i < n; ++i) {
        res += array[i];
    }
    return res;
}