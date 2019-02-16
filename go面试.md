## len() 和 cap()

* len() 可以用来查看数组或 slice 的长度
* cap() 可以用来查看数组或 slice 的容量
* 数组由于长度固定不变，因此 len(arr) 和 cap(arr) 的输出永远相同。在 slice 中，len(sli）表示可见元素有几个，而 cap(sli) 表示所有元素有几个

## make

make 仅仅作用于 map, slice, 以及 channel，不会返回指针

## map 取键值

map 取键值，如果某个键不存在，一般会返回 0 值。那么如果将这种场景和真正的 0 值区分：
`seconds, ok := timezone['tk']`
