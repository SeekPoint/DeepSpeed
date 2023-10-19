// https://blog.csdn.net/cabinriver/article/details/8960119

//#include <string>
//string str ('/home/amd00/yk_repo/ds/DeepSpeed/csrc');
//cutlen = str.size();


#define __output(...) \
    printf(__VA_ARGS__);

#define __format(__fmt__) "%s(%d)-<%s>: " __fmt__ "\n"

#define debuginfo(__fmt__, ...) __output(__format(__fmt__), __FILE__, __LINE__, __FUNCTION__, ##__VA_ARGS__);
//4、使用到的宏
//　　1) __VA_ARGS__   是一个可变参数的宏，这个可宏是新的C99规范中新增的，目前似乎gcc和VC6.0之后的都支持（VC6.0的编译器不支持）。宏前面加上##的作用在于，当可变参数的个数为0时，这里的##起到把前面多余的","去掉的作用。
//　　2) __FILE__    宏在预编译时会替换成当前的源文件名
//　　3) __LINE__   宏在预编译时会替换成当前的行号
//　　4) __FUNCTION__   宏在预编译时会替换成当前的函数名称



// 去掉宏__FILE__的路径 https://blog.coderhuo.tech/2017/04/14/__FILE__strip_path/
