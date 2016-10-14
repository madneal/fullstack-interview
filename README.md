# front-end-interview-process

我觉得每一个前端开发工程师上辈子都是断翼的天使。最近一直在找前端开发工程师的岗位，也是很艰辛，希望能够通过这个纪录自己在前端面试过程中遇到的困难或者自己在学习上遇到的问题。

# HTML 篇

# CSS 篇

1. 通过css实现一个三列布局，左右宽度固定，中中间的内容宽度自适应，并且两边元素的高度随着中间元素的高度变化而变化

# javascript篇

1. 通过js实现一个函数找到字符串中出现次数最多的字符
2. 如何实现深度克隆
3. 如何解析一个url，http://baidu.com?name=123&age=123#secion解析query的内容
4. 有如下一份伪json文件：

```javascript
{
     "name": "qiniu",
     "location": "shanghai", // in shanghai
     "tags": ["#golang", "//tech", "/\\\"//enthusiasm\"/"], // //manytags
     // a comment line
     "property": {
       // these are properties
       "products":["/storage//", "cdn", "/data/processing/", "////live///"] //4 products
     }
}
```

现在要实现一个解析器，将上述文件转化为合法的json文件，即去掉注释，上述文件转化为：

```javascript
{
     "name": "qiniu",
     "location": "shanghai",
     "tags": ["#golang", "//tech", "/\\\"//enthusiasm\"/"], 
     "property": {
       "products":["/storage//", "cdn", "/data/processing/", "////live///"] 
     }
}
```



# 性能篇

1. 为什么要对js文件进行打包（新美大面试题）



# 框架

1. jquey是如何实现选择器的（新美大面试题）
2. undescore里面的template方法（新美大面试题）



# 其它问题

1. 有没有阅读过什么框架的源码（新美大面试题）