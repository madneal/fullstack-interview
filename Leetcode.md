# Leetcode

## [Binary Trees With Factors](https://leetcode.com/problems/binary-trees-with-factors/description/)

```java
public int numFactoredBinaryTrees(int[] A) {
  long res = 0L;
  long mod = (long) Math.pow(10, 9) + 7;
  Arrays.sort(A);
  HashMap<Integer, Long> dp = new HashMap<Integer, Long>();
  for (int i = 0; i < A.length; ++i) {
    dp.put(A[i], 1L);
    for (int j = 0; j < i; ++j) {
      if (A[i] % A[j] == 0 && dp.containsKey(A[i] / A[j])) {
        dp.put(A[i], (dp.get(A[i]) + dp.get(A[j]) * dp.get(A[i] / A[j])) % mod);
      }
    }
  }
  for (long v: dp.values()) {
    res = (res + v) % mod;
  }
  return (int) res;
}
```

## [Count Complete Tree Nodes](https://leetcode.com/problems/count-complete-tree-nodes/description/)

```java
public int countNode(TreeNode root) {
  int nodes = 0;
  int h = height(root);
  while (root != null) {
    if (height(root.right) == h -1) {
      nodes += 1 << h;
      root = root.right;
    } else {
      nodes += 1 << (h - 1);
      root = root.left;
    }
    h--;
  }
  return nodes;
}

private int height(TreeNode root) {
  return root == null ? -1 : height(root.left) + 1;
}
```

## [Remove Linked List Elements](https://leetcode.com/problems/remove-linked-list-elements/description/)

```java
public ListNode removeElements(ListNode head, int val) {
  if (head == null) {
    return head;
  }
  head.next = removeElements(head.next, val);
  return head.val == val ? head.next : head;
}
```

## [Single Number III](https://leetcode.com/problems/single-number-iii/description/)

```java
public int[] singleNumber(int[] nums) {
  int diff = 0;
  for (int num: nums) {
    diff ^= num;
  }
  diff &= -diff;
  int[] result = {0, 0};
  for (int num: nums) {
    if ((num & diff) == 0) {
      result[0] ^= num;
    } else {
      result[1] ^= num;
    }
  }
  return result;
}
```

## [Convert BST to Greater Tree](https://leetcode.com/problems/convert-bst-to-greater-tree/description/)

```java
int sum = 0;
public TreeNode convertBST(TreeNode root) {
  convert(root);
  return root;
}

private void convert(TreeNode cur) {
  if (cur == null) {
    return;
  }
  convert(cur.right);
  cur.val += sum;
  sum = cur.val;
  convert(cur.left);
}
```

## [Base 7](https://leetcode.com/problems/base-7/description/)

```java
public String convertToBase7(int num) {
  if (num < 0) {
    return "-" + convertToBase7(-num);
  }
  if (num < 7) {
    return "" + num;
  }
  return convertToBase7(num / 7) + num % 7;
}
```

## [Find Mode in Binary Search Tree](https://leetcode.com/problems/find-mode-in-binary-search-tree/description/)

```java
int cnt = 1;
int max = 0;
Integer prev = null;
public int[] findMode(TreeNode root) {
  if (root == null) {
    return new int[0];
  }
  List<Integer> list = new ArrayList<Integer>();
  findMax(root, list);
  int[] result = new int[list.size()];
  for (int i = 0; i < list.size(); i++) {
    result[i] = list.get(i);
  }
  return result;
}

private void findMax(TreeNode root, List<Integer> list) {
  if (root == null) {
    return;
  }
  findMax(root.left, list);
  if (prev != null) {
    if (root.val == prev) {
      cnt++;
    } else {
      cnt = 1;
    }
  }
  if (cnt == max) {
    result.add(root.val);
  } else if (cnt > max) {
    max = cnt;
    result.clear();
    result.add(root.val);
  }
  prev = root.val;
  findMax(root.right, list);
}
```

## [Longest Harmonious Subsequence](https://leetcode.com/problems/longest-harmonious-subsequence/description/)

```java
public int findLHS(int[] nums) {
  int max = 0;
  HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();
  Arrays.sort(nums);
  for (int num: nums) {
    map.put(num, map.getOrDefault(num, 0) + 1);
    if (map.containsKey(num - 1)) {
      max = Math.max(max, map.get(num - 1) + map.get(num));
    }
  }
  return max;
}
```

## [Linked List Cycle](https://leetcode.com/problems/linked-list-cycle/description/)
```java
public boolean hasCycle(ListNode head) {
  if (head == null) {
    return false;
  }
  ListNode slow = head;
  ListNode fast = head.next;
  while (slow != null && fast != null && fast.next != null && fast != slow) {
    slow = slow.next;
    fast = fast.next.next;
  }
  return fast != null && slow == fast;
```

```java
public boolean hasCycle(ListNode head) {
  HashMap<ListNode, Integet> hash = new HashMap<ListNode, Integer>();
  int i = 0;
  while (head != null) {
    if (hash.containsKey(head)) {
      return true;
    } else {
      hash.put(head, i);
      i++;
    }
    head = head.next;
  }
  return false;
}
```

## [Reverse Pairs](https://leetcode.com/problems/reverse-pairs/description/)

```java
  public int reversePairs(intp[ nums) {
    return mergeSort(nums, 0, nums.length - 1);
  }
  
  private int mergeSort(int[] nums, int s, int e) {
    if (s >= e) {
      return 0;
    }
    int mid = s + (e -s) / 2;
    int cnt = mergeSort(nums, s, mid) + mergeSort(nums, mid + 1, e);
    for (int i = s, j = mid + 1; i <= mid; i++) {
      while (j <= e && nums[i] / 2.0 > nums[j]) {
        j++;
      }
      cnt += j - (mid + 1);
    }
    Arrays.sort(nums, s, e + 1);
    return cnt;
  }
}
```

## [Symmetric Tree](https://leetcode.com/problems/symmetric-tree/description/)

```java
public boolean isSymmetric(TreeNode root) {
  if (root == null) {
    return true;
  }
  return symmetric(root.left, root.right);
}

private boolean symmetric(TreeNode left, TreeNode right) {
  if (left == null || right == null) {
    return left == right;
  }
  if (left.val != right.val) {
    return false;
  }
  return symmetric(left.left, right.right) && symmetric(left.right, right.left);
}
```

## [Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements/description/)

```javascript
var topKFrequent = function(nums, k) {
  var hash = {};
  var result = [];
  for (var i = 0; i < nums.length; i++) {
    if (!hash[nums[i]]) {
      hash[nums[i]] = 1;
    } else {
      hash[nums[i]]++;
    }
  }
  var listBucket = [];
  for (var key in hash) {
    var frequence = hash[key];
    if (!listBucket[frequence]) {
      listBucket[frequence] = [];
    }
    listBucket[frequence].push(key);
  }
  
  for (var i = listBucket.length; i >= 0; i--) {
    if (listBucket[i]) {
      var len = listBucket[i].length;
      for (var j = 0; j < len; j++) {
        if (result.length < k) {
	  result.push(+listBucket[i][j]);
	} else {
	  break;
	 }
        } 
      }
    
  }
  return result;
}
 ```
       

## [Shortest Distance to a Character](https://leetcode.com/problems/shortest-distance-to-a-character/description/)

```java
public int[] shortestToChar(String S, Char C) {
  List<Integer> aux = new ArrayList<Integer>();
  for (int i = 0; i < s.length(); i++) {
    if (S.charAt(i) == C) {
      aux.add(i);
    }
  }
  int[] res = new int[S.length()];
  for (int = 0; i < S.length(); i++) {
    res[i] = (int) Math.abs(i - binSearch(aux, i));
  }
  return res;
}

private int binSearch(List<Integer> a, int key) {
  int l = 0;
  int h = a.size() - 1;
  if (key < a.get(l)) {
    return a.get(l);
  }
  if (key > a.get(h)) {
    return a.get(h);
  }
  while (l <= h) {
    int m = l + (h -l) / 2;
    if (a.get(m) == key) {
      return a.get(m);
    } else if (key < a.get(m)) {
      h = m - 1;
    } else {
      l = m + 1;
    }
  }
  return a.get(l) - key < key - a.get(h) ? a.get(l) : a.get(h);
  }
}
```

## [Generate Parentheses](https://leetcode.com/problems/generate-parentheses/description/)

```java
public List<String> generateParenthesis(int n) {
  List<String> result = new ArrayList<String>();
  backtrack(result, "", 0, 0, n);
  return result;
}

private void backtrack(List<String> result, String str, int open, int close, int max) {
  if (str.length() == 2 * max) {
    result.add(str);
    return;
  }
  
  if (open < max) {
    backtrack(result, str + "(", open + 1, close, max);
  }
  
  if (close < open) {
    backtrack(result, str + ")", open, close + 1, max);
  }
}
```

## [Beautiful Arrangement II](https://leetcode.com/problems/beautiful-arrangement-ii/description/)

```java
public int[] constructArray(int n, int k) {
  int res = new int[n];
  int left = 1;
  int right = n;
  for (int i = 0; i < n; i++) {
    res[i] = K % 2 != 0 ? left++ : right--;
    if (k > 1) {
      k--;
    }
  }
  return res;
}
```

## [Delete Operation for Two Strings](https://leetcode.com/problems/delete-operation-for-two-strings/description/)

```java
public int minDistance(String word1, String word2) {
  int m = word1.length();
  int n = word2.length();
  int[][] dp = new int[m + 1][n + 1];
  for (int i = 0; i <= m; i++) {
    for (int j = 0; j <= n; j++) {
      if (i == 0 || j == 0) {
        dp[i][j] = 0;
      } else {
        if (word1.charAt(i - 1) == word2.charAt(j -1)) {
	  dp[i][j] = 1 + dp[i - 1][j - 1];
	} else {
	  dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
	}
      }
    }
  }
  return m + n - 2 * dp[m][n];
}
```

## [Number of Longest Increasing Subsequence](https://leetcode.com/problems/number-of-longest-increasing-subsequence/description/)

```java
public int findNumberOfLTS(int[] nums) {
  int n = nums.length;
  int res = 0;
  int maxLen = 0;
  int[] len = new int[n];
  int[] cnt = new int[n];
  for (int i = 0; i < n; i++) {
    len[i] = cnt[i] = 1;
    for (int j = 0; j < i; j++) {
      if (nums[i] > nums[j]) {
        if (len[i] == len[j] + 1) {
	  cnt[i] += cnt[j];
	} else if (len[i] < len[j] + 1) {
	  len[i] = len[j] + 1;
	  cnt[i] = cnt[j];
	}
       }
     }
     if (maxLen == len[i]) {
       res += cnt[i];
     } else if (maxLen < len[i]) {
       maxLen = len[i];
       res = cnt[i];
     }
   }
   return res;
 }
 ```

## [Maximum Length of Repeated Subarray](https://leetcode.com/problems/maximum-length-of-repeated-subarray/description/)

```java
public int findLength(int[] A, int[] B) {
  int m = A.length;
  int n = B.length;
  int max = 0;
  int[][] dp = new int[m + 1][n + 1];
  for (int i = 0; i <= m; i++) {
    for (int j = 0; j <= n; j++) {
      if (i == 0 || j == 0) {
        dp[i][j] = 0;
      } else {
        if (A[i - 1] == B[j -1]) {
	  dp[i][j] = 1 + dp[i - 1][j - 1];
	  max = Math.max(max, dp[i][j]);
	}
      }
    }
  }
  return max;
}
```

## [Invert Binary Tree](https://leetcode.com/problems/invert-binary-tree/description/)

```java
public TreeNode invertTree(TreeNode root) {
  if (root == null) {
    return null;
  } else {
    TreeNode tem = root.left;
    root.left = root.right;
    root.right = tem;
  }
  root.left = invertTree(root.left);
  root.right = invertTree(root.right);
  return root;
}
```

## [Maximal Square](https://leetcode.com/problems/maximal-square/description/)

```java
public int maximalSquare(char[][] matrix) {
  if (matrix.length == 0) {
    return 0;
  }
  int m = matrix.length;
  int n = matrix[0].length;
  int result = 0;
  int[][] b = new int[m + 1][n + 1];
  for (int i = 1; i <= m; i++) {
    for (int j = 1; j <= n; j++) {
      if (matrix[i - 1][j - 1] == '1') {
        b[i][j] = Math.min(Math.min(b[i][j - 1], b[i -1][j]), b[i - 1][j -1]);
	result = Math.max(result, b[i][j]);
      }
    }
  }
  return result * result;
}
```

## [Copy List with Random Pointer](https://leetcode.com/problems/copy-list-with-random-pointer/description/)

```java
public RandomListNode copyRandomList(RandomListNode head) {
  if (head == null) {
    return head;
  }
  RandomListNode c = head;
  while (c != null) {
    RandomListNode next = c.next;
    c.next = new RandomListNode(c.label);
    c.next.next = next;
    c = next;
  }
  
  c = head;
  while (c != null) {
    if (c.random != null) {
      c.next.random = c.random.next
    }
    c = c.next.next;
  }
  
  c = head;
  RandomListNode copyHead = c.next;
  RandomListNode copy = copyHead;
  while (copy.next != null) {
    c.next = c.next.next;
    c = c.next;
    
    copy.next = copy.next.next;
    copy = copy.next;
  }
  c.next = c.next.next;
  return copyHead;
} 
```

## [Longest Palindrome](https://leetcode.com/problems/longest-palindrome/description/)

```java
public int longestPalindrome(String s) {
  if (s == null || s.length() == 0) {
    return 0;
  }
  int count = 0;
  HashSet<Character> hs = new HashSet<Character>();
  for (int i = 0; i < s.length(); i++) {
    if (hs.contains(s.charAt(i)) {
      hs.remove(s.charAt(i));
      count++;
    } else {
      hs.add(s.charAt(i));
    }
  }
  if (hs.isEmpty()) {
    return count * 2;
  } else {
    return count * 2 + 1;
  }
}
```

## [Longest Continuous Increasing Subsequence](https://leetcode.com/problems/longest-continuous-increasing-subsequence/description/)

```javascript
var findLengthOfLCIS = function(nums) {
  var res = 0;
  var cnt = 0;
  for (var i = 0; i < nums.length; i++) {
    if (i == 0 || nums[i - 1] < nums[i]) {
      res = Math.max(res, ++cnt);
    } else {
      cnt = 1;
    }
  }
  return res;
}
```

## [Find All Duplicates in an Array](https://leetcode.com/problems/find-all-duplicates-in-an-array/description/)

```go 
func findDuplicates(nums []int) []int {
  result := make([]int, 0)
  for _, num := range nums {
    index := int(math.Abs(float64(num))) - 1
    if nums[index] < 0 {
      result = append(result, index + 1)
    }
    nums[index] = -nums[index]
  }
  return result
}
```
   
## [Construct Binary Tree from Preorder and Inorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/description/)

```go 
func buildTree(preorder []int, inorder []int) *TreeNode {
  if len(preorder) != len(inorder) {
    return nil
  }
  return build(preorder, inorder, 0, len(preorder) - 1, 0, len(inorder) -1)
}

func build(preorder, inorder []int, prelow, prehigh, inlow, inhigh int) *TreeNode {
  if prelow > prehigh || inlow > inhigh {
    return nil 
  }
  var root *TreeNode = &TreeNode{preorder[prelow], nil, nil}
  inorderRoot := inlow
  for i := inlow; i <= inhigh; i++ {
    if inorder[i] == root.Val {
      inorderRoot = i
      break
    }
  }
  inorderLen := inorderRoot - inlow
  root.Left = build(preorder, inorder, prelow + 1, prelow + inorderLen, inlow, inorderRoot - 1)
  root.Right = build(preorder, inorder, prelow + inorderLen + 1, prehigh, inorderRoot + 1, inhigh)
  return root
}
```

## [Partition List](https://leetcode.com/problems/partition-list/description/)

```java
public ListNode partition(ListNode head, int x) {
  ListNode node0 = new ListNode(0);
  ListNode node1 = new ListNode(0);
  ListNode headSmaller = node0;
  ListNode headGreater = node1;
  while (head != null) {
    if (head.val < x) {
      headSmaller.next = head;
      headSmaller = head;
    } else {
      headGreater.next = head;
      headGreater = head;
    }
  }
  headGreater.next = null;
  headSmaller.next = node1.next;
  return node0.next;
}
```

## [Find All Anagrams in a String](https://leetcode.com/problems/find-all-anagrams-in-a-string/description/)

```javascript
var findAnagrams = function(s, p) {
  var hash = {};
  
  for (var i = 0; i < p.length; i++) {
    if (!hash[p[i]) {
      hash[p[i]] = 1;
    } else {
      hash[p[i]]++;
    }
  }
  
  var left = 0;
  var right = 0;
  var count = p.length;
  var result = [];
  
  while (right < s.length) {
    if (hash[p[right] >= 1) {
      count--;
    }
    hash[p[right]]--;
    right--;
    
    if (count === 0) {
      result.push(left);
    }
    
    if (right - left === p.length) {
      if (hash[p.left] >= 0) {
        count++;
      }
      hash[p[right]]++;
      left++;
    }
  }
  return result;
}
```

## [Repeated DNA Sequences](https://leetcode.com/problems/repeated-dna-sequences/description/)

```javascript
var findRepeatedDnaSequences = function(s) {
  var hash = {};
  var result = [];
  for (var i = 0; i < s.length; i++) {
    var key = s.substring(i, i + 10);
    if (!hash[key]) {
      hash[key] = 1;
    } else {
      if (hash[key] < 2) {
        result.push(key);
      }
      hash[key]++;
    }
  }
  return result;
}
```


## [Same Tree](https://leetcode.com/problems/same-tree/description/)

```go 
func isSameTree(p *TreeNode, q *TreeNode) {
  if p == nil && q == nil {
    return true
  }
  
  if p == nil || q == nil {
    return false
  }
  
  if p.Val == q.Val {
    return isSameTree(p.Left, q.Left) && isSameTree(p.Right, q.Right)
  }
  return false
}
```

## [Convert a Number to Hexadecimal](https://leetcode.com/problems/convert-a-number-to-hexadecimal/description/)

```java
public String toHex(int num) {
  if (num == 0) {
    return "0";
  }
  char[] map = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};
  String result = "";
  while (num != 0) {
    result = map[num & 15] + result;
    num = (num >>> 4);
  }
  return num;
}
```

## [Contiguous Array](https://leetcode.com/problems/contiguous-array/description/)

```go 
func findMaxLength(nums []int) int {
  for i := 0; i < len(nums); i++ {
    if nums[i] == 0 {
      nums[i] = -1
    }
  }
  sumToIndex := make(map[int]int, 0)
  sum := 0
  max := 0
  
  for i := 0; i < len(nums); i++ {
    sum += nums[i]
    if _, ok := sumToIndex[sum]; ok {
      if max < i - sumToIndex[sum] {
        max = i - sumToIndex[sum]
      }
    } else {
      sumToIndex[sum] = i
    }
  }
  return max
}
```

## [Jump Game II](https://leetcode.com/problems/jump-game-ii/description/)

```java
public int jump(int[] nums) {
  int step = 0;
  int lastMax = 0; 
  int currentMax = 0;
  
  for (int i = 0; i < nums.length - 1; i++) {
    currentMax = Math.max(currentMax, i + nums[i]);
    if (i == lastMax) {
      step++;
      currentMax = lastMax;
    }
  }
  return step;
}
```
    

## [Binary Tree Level Order Traversal II](https://leetcode.com/problems/binary-tree-level-order-traversal-ii/description/)

```java
public List<List<Integer>> levelOrderBottom(TreeNode root) {
  List<List<Integer>> result = new ArrayList<List<Integer>>();
  if (root == null) {
    return result;
  }
  List<TreeNode> currentLevel = new ArrayList<TreeNode>();
  currentLevel.add(root);
  
  while (currentLevel.size() != 0) {
    List<Integer> valueList = new ArrayList<Integer>();
    List<TreeNode> nextLevel = new ArrayList<TreeNode();
    
    for (int i = 0; i < currentLevel.size(); i++) {
      TreeNode node = currentLevel.get(i);
      valueList.add(node.val);
      if (node.left != null) {
        nextLevel.add(node.left);
      }
      
      if (node.right != null) {
        nextLevel.add(node.right);
      }
    }
    result.add(valueList);
    currentLevel = nextLevel;
  }
  Collections.reverse(result);
  return result;
}
```

## [Binary Tree Inorder Traversal](https://leetcode.com/problems/binary-tree-inorder-traversal/description/)

```javascript
/**
 * Definition for a binary tree node.
 * function TreeNode(val) {
 *     this.val = val;
 *     this.left = this.right = null;
 * }
 */
/**
 * @param {TreeNode} root
 * @return {number[]}
 */
var inorderTraversal = function(root) {
  var stack = [];
  var result = [];
  var cur = root;
  while (cur !== null || stack.length !== 0) {
    if (cur !== null) {
      stack.push(cur);
      cur = cur.left;
    } else {
      cur = stack.pop();
      result.push(cur.val);
      cur = cur.right;
    }
  }
  return result;
}
```

## [Minimum Depth of Binary Tree](https://leetcode.com/problems/minimum-depth-of-binary-tree/description/)

```go
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
func minDepth(root *TreeNode) int {
  if root == nil {
    return 0
  }
  
  left := minDepth(root.Left)
  right := minDepth(root.Right)
  
  if left == 0 || right == 0 {
    return left + right + 1
  } else {
    if left < right {
      return left + 1
    } else {
      return right + 1
    }
  }
}
```

## [Valid Parentheses](https://leetcode.com/problems/valid-parentheses/description/)

```javascript
/**
 * @param {string} s
 * @return {boolean}
 */
var isValid = function(s) {
  var stack = [];
  
  for (var i = 0; i < s.length; i++） {
    var ele = s[i];
    
    if (ele === '(') {
      stack.push(')');
    } else if (ele === '{') {
      stack.push('}');
    } else if (ele === '[') {
      stack.push(']');
    } else if (stack.length === 0 || stack.pop() !== ele) {
      return false;
    }
    return stack.length === 0;
}
```

## [Reverse Words in a String III](https://leetcode.com/problems/reverse-words-in-a-string-iii/description/)

```javascript
/**
 * @param {string} s
 * @return {string}
 */
var reverseWords = function(s) {
  var result = "";
  var words = s.split(' ');
  for (var i = 0; i < words.length; i++) {
    var word = words[i];
    if (i === 0) {
      result += word.split('').reverse().join('');
    } else {
      result ++ ' ' + word.split('').reverse().join('');
    }
  }
  return result;
}
```

## [Set Matrix Zeroes](https://leetcode.com/problems/set-matrix-zeroes/description/)

```go
func setZeros(matrix [][]int) {
  n := len(matrix)
  m := len(matrix[0])
  col := 1
  
  for i := 0; i < n; i++ {
    if matrix[i][0] == 0 {
      col = 0
    }
    for j := 1; j < m; j++ {
      if matrix[i][j] == 0 {
        matrix[i][0] = 0
	matrix[0][j] = 0
      }
    }
  }
  
  for i := n - 1; i >= 0; i-- {
    for j := m - 1; j >= 1; j-- {
      if matrix[i][0] == 0 || matrix[0][j] == 0 {
        matrix[i][j] = 0
      }
    }
    if col == 0 {
      matrix[i][0] = 0
    }
  }
}
```
    


## [Odd Even Linked List](https://leetcode.com/problems/odd-even-linked-list/description/)

```go
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func oddEvenList(head *ListNode) *ListNode {
  if head != nil {
    odd := head
    even := head.Next
    evenHead := even
    for (even != nil && even.Next != nil) {
      odd.Next = odd.Next.Next
      even.Next = even.Next.Next
      odd = odd.Next
      even.Next = even.Next
    }
    odd.Next = evenHead
  }
  return head
}
```



## [Single Number II](https://leetcode.com/problems/single-number-ii/description/)

```go
func singleNumber(nums []int) int {
	ones := 0
	twos := 0
	for i := 0; i < len(nums); i++ {
		ones = (ones ^ nums[i]) & (^twos)
		twos = (twos ^ nums[i]) & (^ones)
	}
	return ones
}
```

## [Word Search](https://leetcode.com/problems/word-search/description/)

```go
func exist(board [][]byte, word string) {
	w := []byte(word)
	for y := 0; y < len(board); y++ {
		for x :=0; x < len(board[0]); x++ {
			if exist1(board, w, x, y, 0) {
				return true
			}
		}
	}
	return false
}

func exist1(board [][]byte, word []byte, x, y, index int) {
	if index == len(word) {
		return true
	}
	
	if x < 0 || y < 0 || x = len(board[0]) || y == len(board) {
		return false
	}
	
	if board[y][x] != word[index] {
		return false
	}
	
	board[y][x] = '*'
	result = exist1(board, word, x + 1, y, index + 1) ||
		exist1(board, word, x - 1, y, index + 1) ||
		exist1(board, word, x, y + 1, index + 1) ||
		exist1(board, word, x, y - 1, index + 1)
	board[y][x] = word[index]
	return result
}
```

## [Partition to K Equal Sum Subsets](https://leetcode.com/problems/partition-to-k-equal-sum-subsets/description/)

```go
func canPartitionKSubsets(nums []int, k int) {
	sum := 0
	for _, num : range nums {
		sum += num
	}
	if k <= 0 || sum % k != 0 {
		return false
	}
	visited := make([]int, len(nums))
	return canPartion(nums, visited,
}

func canPartion(nums []int, visited []int, startIndex, k, curSum, curNum, target int) {
	if k == 1 {
		return true
	}
	
	if curSum == target && curNum > 0 {
		return canPartion(nums, visited, 0, k - 1, 0, 0, target)
	}
	
	for i := startIndex; i < len(nums); i++ {
		if visited[i] == 0 {
			visited[i] = 1
			curNum++
			
			if (canPartion(nums, visited, i + 1, k, curSum + nums[i], curNum, target) {
				return true
			}
			visited[i] = 0
		}
	}
	return false
}
```


## [Combinations](https://leetcode.com/problems/combinations/description/)

```java
public List<List<Integet>> combine(int n, int k) {
	List<List<Integer>> result = new ArrayList<List<Integer>>();
	List<Integer> temp = new ArrayList<Integer>();
	dfs(result, temp, k, 1, n);
	return result;
}

public dfs(List<List<Integer>> result, List<Integer> temp, int k, int from, int to) {
	if (k == 0) {
		result.add(new ArrayList<Integer>(temp);
		return;
	}
	
	for (int i = from; i <= to; i++) {
		temp.add(i);
		dfs(result, temp, k - 1, from + 1, to + 1);
		temp.remove(temp.size() - 1);
	}
}
```

## [Largest Number](https://leetcode.com/problems/largest-number/description/)

```java
public String largestNumber(int[] nums) {
	String[] arr = new String[nums.length];
	for (int i = 0; i < nums.length; i++) {
		arr[i] = String.valueOf(nums[i]);
	}
	Arrays.sort(arr, (s1, s2) -> (s2 + s1).compareTo(s1 + s2));
	if (arr[0].charAt[0] == '0') {
		return '0';
	}
	StringBuilder sb = new StringBuilder();
	for (String s: arr) {
		sb.append(s);
	}
	return sb.toString();
}
```

## [Single Element in a Sorted Array](https://leetcode.com/problems/single-element-in-a-sorted-array/description/)

```go
func singleNonDuplicate(nums []int) int {
	n := len(nums)
	low := 0
	high := n / 2
	for (low < high) {
		m := (low + high) / 2
		if nums[2 * m] != nums[2 * m + 1] {
			high = m
		} else {
			low = m + 1
		}
	}
	return nums[2 * low]
}
```

## [Coin Change](https://leetcode.com/problems/coin-change/description/)

```go
func coninChnage([]int coins, amount int) {
	if amount < 1 {
		return 0
	}
	dp := make(int[], amount + 1)
	
	coins = sotr.Ints(coins)
	sum := 1
	
	for (sum <= amount) {
		min := -1
		for _, coin := range coins {
			if sum < coin {
				break
			}
			temp = dp[sum - coin] + 1
			if min < 0 {
				min = temp
			} else {
				min = int(math.Min(float64(temp), float64(min)))
			}
		}
		dp[sum] = min
		sum ++
	}
	return dp[amount]
}
```

## [Add Binary](https://leetcode.com/problems/add-binary/description/)

```go
func max(a,b int) int {
	if a > b {
		return a
	} else {
		return b
	}
}

func addBinary(a string, b string) string {
	m := len(a)
	n := len(b)
	maxLen := max(m, n)
	carry := 0
	res := ""
	
	for i := 0; i < maxLen; i++ {
		var p int 
		var q int
		if i < m {
			p, _ = strconv.Atoi(string(a[m - i - 1])) 
		} else {
			p = 0
		}
		
		if i < n {
			q, _ = strconv.Atoi(string(b[n -i -1]))
		} else {
			q = 0
		}
		temp := p + q + carry
		res = strconv.Itoa(temp % 2) + res
		if temp > 1 {
			carry = 1
		} else {
			carry = 0
		}
	}
	
	if carry == 0 {
		return res
	} else {
		return "1" + res
	}
}

```

## [Triangle](https://leetcode.com/problems/triangle/description/)

```go
func minimumTotal([][]int triangle) int {
	rowNum := len(triangle)
	var minpath int
	for i := rowNum - 2; i >= 0; i-- {
		nextRow = triangle[i + 1]
		for j := o; i <=i; j++ {
			minpath = int(math.Min(float64(nextRow[j]), float64(nextRow[j + 1))) + triangle[i][j]
			trinangle[i][j] = minpath
		}
	}
	return triangle[0][0]
}
```

## 1.Serialize and deserialize binary tree

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
public class Codec {
    private static final String NN = "x";
    private static final String splitor = ",";

    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        StringBuilder sb = new StringBuilder();
        buildString(root, sb);
        return sb.toString();
    }
    
    private void buildString(TreeNode root, StringBuilder sb) {
        if (root == null) {
            sb.append(NN).append(splitor);
        } else {
            sb.append(root.val).append(splitor);
            buildString(root.left, sb);
            buildString(root.right, sb);
        }
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        Deque<String> nodes = new LinkedList<>();
        nodes.addAll(Arrays.asList(data.split(splitor)));
        return buildTree(nodes);
    }
    
    private TreeNode buildTree(Deque<String> nodes) {
        String val = nodes.remove();
        if (val.equals(NN)) {
            return null;
        } else {
            TreeNode node = new TreeNode(Integer.valueOf(val));
            node.left = buildTree(nodes);
            node.right = buildTree(nodes);
            return node;
        }
    }
}
```

## 2.Intersection of two arrays

```java
public class Solution {
    public int[] intersection(int[] nums1, int[] nums2) {
        HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();
        List<Integer> result = new ArrayList<Integer>();
        int len1 = nums1.length;
        int len2 = nums2.length;
        for (int i = 0; i < len1; i ++) {
            if (map.containsKey(nums1[i])) {
                continue;
            } else {
                map.put(nums1[i], 1);
            }
        }
        for (int j = 0; j < len2; j ++) {
            if (map.containsKey(nums2[j]) && map.get(nums2[j]) <= 1) {
                result.add(nums2[j]);
                map.put(nums2[j], map.get(nums2[j]) + 1);
            }
        }
        int length = result.size();
        int[] res = new int[length];
        for (int k = 0; k < length; k++) {
            res[k] = result.get(k);
        }
        return res;      
    }
}
```

## 3.Intersection of two arrays II

```java
public class Solution {
    public int[] intersect(int[] nums1, int[] nums2) {
        HashMap<Integer, Integer> map = new HashMap<>();
        List<Integer> result = new ArrayList<Integer>();
        int len1 = nums1.length;
        int len2 = nums2.length;
        for (int i = 0; i < len1; i ++) {
            if (map.containsKey(nums1[i])) {
                map.put(nums1[i], map.get(nums1[i]) + 1);
            } else {
                map.put(nums1[i], 1);
            }
        }
        for (int j = 0; j < len2; j ++) {
            if (map.containsKey(nums2[j]) && map.get(nums2[j]) > 0) {
                result.add(nums2[j]);
                map.put(nums2[j], map.get(nums2[j]) - 1);
            } else {
                continue;
            }
        }
        int length = result.size();
        int[] res = new int[length];
        for (int i = 0; i < length; i ++) {
            res[i] = result.get(i);
        }
        return res;
    }
}
```

## 4.Third maximum number

```java
public class Solution {
    public int thirdMax(int[] nums) {
    long max = Long.MIN_VALUE;
    mid = max, min = max;
    for (int ele : nums) {
      if (ele > max) {
        min = mid;
        mid = max;
        max = ele;
      } else if (max > ele && ele > mid) {
        min = mid;
        mid = ele;
      } else if (mid > ele && ele > min) {
        min = ele;
      }
    }
    return (int)(min != Long.MIN_VALUE) ? min : max;
    }
}
```

## 5.Reverse vowels of a string

```java
public class Solution {
    public String reverseVowels(String s) {
        if (s == null || s.length() == 0) {
            return s;
        }
        String vowels = "aeiouAEIOU";
        char[] chars = s.toCharArray();
        int start = 0;
        int end = s.length() - 1;
        while (start < end) {
            while (start < end && !vowels.contains(chars[start]+"")) {
                start++;
            }
            while ( start < end && !vowels.contains(chars[end]+"")) {
                end--;
            }
            char temp = chars[start];
            chars[start] = chars[end];
            chars[end] = temp;
            start++;
            end--;
        }
        return new String(chars);
    }
}
```

## 6.Missing number

```java
public class Solution {
    public int missingNumber(int[] nums) {
        int res = nums.length;
        for (int i = 0;i < nums.length;i++) {
            res ^= i;
            res ^= nums[i];
        }
        return res;
    }
}
```

## 7.Path sum

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
public class Solution {
    public boolean hasPathSum(TreeNode root, int sum) {
        if (root == null) {
            return false;
        }
        if (root.left == null && root.right == null && root.val - sum == 0) {
            return true;
        }
        return hasPathSum(root.left, sum - root.val) || hasPathSum(root.right, sum - root.val);
        
    }
}
```

## 8.Path sumII

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
public class Solution {
    public List<List<Integer>> pathSum(TreeNode root, int sum) {
        List<List<Integer>> result = new ArrayList<>();
        List<Integer> list = new ArrayList<>();
        pathSum(root, sum, result, list);
        return result;
    }
    
    private void pathSum(TreeNode root, int sum, List<List<Integer>> result, List<Integer> list) {
        if (root == null) {
            return;
        }
        list.add(root.val);
        if (root.left == null && root.right == null && sum - root.val == 0) {
            result.add(new ArrayList(list));
        } else {
            pathSum(root.left, sum - root.val, result, list);
            pathSum(root.right, sum - root.val, result, list);
        }
        list.remove(list.size() - 1);
    }
}
```

## 9.Path sum III

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
public class Solution {
    public int pathSum(TreeNode root, int sum) {
        if (root == null) {
            return 0;
        }
        return findPath(root, sum) + pathSum(root.left, sum) + pathSum(root.right, sum);
    }
    
    private int findPath(TreeNode root, int sum) {
        int res = 0;
        if (root == null) {
            return res;
        }
        if (root.val == sum) {
            res ++;
        }
        res += findPath(root.left, sum - root.val);
        res += findPath(root.right, sum - root.val);
        return res;
    }
}
```

## 10. Sum of left leaves

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
public class Solution {
    public int sumOfLeftLeaves(TreeNode root) {
        int ans = 0;
        if (root == null) {
            return 0;
        } 
        if (root.left != null) {
            if (root.left.left == null && root.left.right == null) {
                ans += root.left.val;
            } else {
                ans += sumOfLeftLeaves(root.left);
            }
        }
        if (root.right != null) {
            ans += sumOfLeftLeaves(root.right);
        }
        return ans;
    }
}
```

## 11.First Unique character in a string

```java
public class Solution {
    public int firstUniqChar(String s) {
        HashMap<Character, Integer> map = new HashMap<>();
        int n = s.length();
        for (int i = 0; i < n; i ++) {
            char ele = s.charAt(i);
            if (map.get(ele) == null) {
                map.put(ele, 1);
            } else {
                map.put(ele, map.get(ele) + 1);
            }
        }
        Iterator it = map.entrySet().iterator();
        for (int j = 0; j < n; j++) {
            if (map.get(s.charAt(j)) == 1) {
                return j;
            }
        }
    return -1;
    }
}
```

## 12. Two sum

```java
public class Solution {
  public int[] twoSum(int[] nums, int target) {
    HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();
    int[] result = new int[2];
    int len = nums.length;
    for (int i = 0; i < len; i ++) {
      if (map.containsKey(target - nums[i])) {
        result[0] = map.get(target - nums[i]);
        result[1] = i;
        return result;
      } else {
        map.put(nums[i], i);
      }
    }
    return result;
  }
}
```

## 13.Two sum II - input array is sorted

```java
public class Solution {
    public int[] twoSum(int[] numbers, int target) {
        int[] indices = new int[2];
        if (numbers == null || numbers.length < 2) {
            return indices;
        }
        int left = 0, right = numbers.length - 1;
        while (left < right) {
            int v = numbers[left] + numbers[right];
            if (v == target) {
                indices[0] = left + 1;
                indices[1] = right + 1;
                break;
            } else if (v > target) {
                right--;
            }
