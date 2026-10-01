\# C++ 猫狗分类推理程序



\## 项目说明

使用 C++ 调用 ONNX Runtime 加载 MobileNetV2 模型，实现猫狗图片分类。

替代 Python + Streamlit 版本，满足老师项目 "using C++" 的要求。



\## 文件结构

\- main.cpp                主程序，ONNX Runtime C API 动态加载

\- mobilenetv2.onnx        模型文件（14MB）

\- stb\_image.h             图片加载

\- stb\_image\_resize.h      resize 到 224×224

\- onnxruntime.dll         ONNX Runtime 动态库

\- catdog\_cpp.exe          编译生成的程序



\## 编译命令

g++ -std=c++17 main.cpp -I"onnxruntime-win-x64-1.20.1\\include" -o catdog\_cpp.exe



\## 运行方法

.\\catdog\_cpp.exe <图片路径>



\## 关键实现细节（避坑记录）



\### 1. ONNX Runtime 使用 C API + 动态加载

\- 包含 onnxruntime\_c\_api.h，不用 C++ 封装头文件

\- 通过 LoadLibraryA 加载 onnxruntime.dll

\- 绕过 ORT\_API\_VERSION 版本宏不一致导致的编译错误



\### 2. 模型实际输入是 NHWC

\- 理论上是 NCHW \[1,3,224,224]，实际导出模型为 \[1,224,224,3]

\- 按模型的真实输入形状准备数据



\### 3. 模型输出已包含 Softmax

\- 输出已经是概率，不能再次 softmax，否则结果被错误抹平



\### 4. 预处理实际方式

\- 使用 pixel / 127.5 - 1.0，取值范围 \[-1, 1]

\- 不是通常的 mean/std 标准化 \[0.485, 0.456, 0.406]

\- 使用 mean/std 会导致分类结果明显错误



\### 5. 概率聚合规则

\- 猫类别索引：281-285

\- 狗类别索引：151-268

\- 猫/狗概率分别求和，较大者为预测类别

\- 最大概率 < 0.5 时输出"其他"



\## 验证结果



| 测试图片 | 实际内容 | 输出类别 | 概率 |

|---|---|---|---|

| test.jpg | 猫 | 其他 | 0.3211 |

| test2.jpg | 狗 | 狗 | 0.6323 |

| test3.jpg | 蛋挞 | 其他 | 0.9749 |



> 注意：图片文件名与实际内容不匹配，已计划重命名为 cat.jpg / dog.jpg / other.jpg。



\## 运行截图

见 E:\\code\\edge\\screenshots\\ 目录

