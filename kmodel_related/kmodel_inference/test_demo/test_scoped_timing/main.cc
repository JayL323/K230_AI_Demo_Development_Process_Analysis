#include <iostream>
#include "scoped_timing.hpp"

using std::cout;
using std::endl;

int main()
{
    // ScopedTiming对象创建时开始计时，销毁时结束计时
    // 当debug_mode>0，销毁时打印计时
    {
        int debug_mode = 1;             // （出作用域）销毁时打印计时
        ScopedTiming st("test 1 :", debug_mode);
        for (int i = 0; i < 50; ++i)
        {
            if (i % 10 == 0 && i != 0)
                cout << endl;
            cout << i << ",";
        }
        cout << endl;
    }
    cout << endl;

    {
        int debug_mode = 0; // （出作用域）销毁时不打印计时
        ScopedTiming st("test 2 :", debug_mode);
        for (int i = 0; i < 10; ++i)
        {
            if (i % 10 == 0 && i != 0)
                cout << endl;
            cout << i << ",";
        }
        cout << endl;
    }

    // 第一个ScopedTiming对象，`debug_mode = 1`出作用域时，打印了test 1部分的耗时；
    // 第二个ScopedTiming对象，`debug_mode = 0`出作用域时，并未打印耗时
    return 0;
}