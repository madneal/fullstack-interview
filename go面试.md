## 调用 defer 错误

```go
package main

import (
	"log"
	"os"
)

func main() {
	for i := 0; i < 5; i++ {
		f, err := os.Open("/path/to/file")
		if err != nil {
			log.Fatal(err)
		}
		defer f.Close()
	}

	zero := 0
	println(1 / zero) 
}
```

程序执行到这里异常退出，那么上面的循环中打开的 5 个文件句柄全部无法泄露

## map 打印

```go
package main
import "fmt"
func main() {
	a := map[int]int{0: 1, 1: 2, 2: 3, 3: 4, 4: 5}
	for _, c := range a {
		fmt.Println(c)
	}
}
```

这段代码的输出结果是什么样的


## 用 slice 模拟栈

```go
package main

import (
	"fmt"
)

var stack []int

func main() {
	for i := 0; i < 5; i++ {
		push(i)
	}
	fmt.Println(stack)
	pop()
	fmt.Println(stack)
	
}

func push(value int) {
	stack = append(stack, value)
}

func pop() int {
	if len(stack) == 0 {
		return -1
	}
	val := stack[len(stack)-1]
	stack = stack[:len(stack)-1]
	return val
}
```

## len() 和 cap()

* len() 可以用来查看数组或 slice 的长度
* cap() 可以用来查看数组或 slice 的容量
* 数组由于长度固定不变，因此 len(arr) 和 cap(arr) 的输出永远相同。在 slice 中，len(sli）表示可见元素有几个，而 cap(sli) 表示所有元素有几个

## make

make 仅仅作用于 map, slice, 以及 channel，不会返回指针

## map 取键值

map 取键值，如果某个键不存在，一般会返回 0 值。那么如果将这种场景和真正的 0 值区分：
`seconds, ok := timezone['tk']`

## 变量初始化

在 Go 语言中，所有变量都被初始化为其零值。对于数值类型，零值是 0；对于字符串类型，零值是空字符串；对于布尔类型，零值是 false；对于指针，零值是 nil。对于引用类型来说，所引用的底层数据结构会被初始化为对应的零值。但是被声明为其零值的引用类型的变量，会返回 nil 作为其值
