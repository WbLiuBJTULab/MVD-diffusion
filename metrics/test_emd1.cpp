#include <iostream>
#include <ATen/cuda/CUDAContext.h>

using namespace std;

int main() {
    c10::Half a(0.5);
    c10::Half b(1.0);

    // 禁用GPU运算符重载后，编译器应该无法识别下面两个运算符，请注意大小写
    c10::Half c = a + b;    // 加法运算符
    bool equal = (a == b);  // 相等运算符

    cout << c << endl;

    return 0;
}