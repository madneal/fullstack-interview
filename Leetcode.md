# Leetcode

## [Most Profit Assigning Work](https://leetcode.com/problems/most-profit-assigning-work/description/)

```java
public int maxProfitAssignment(int[] difficulty, int[] profit, int[] worker) {
  TreeMap<Integer, Integer> tmap = new TreeMap<>();
  for (int i = 0; i < difficulty.length; i++) {
    tmap.put(difficulty[i], Math.max(profit[i],tmap.getOrDefault(difficulty[i], 0)));
  }
  int res = 0;
  int max = 0;
  for (Integer key: tmap.keySet()) {
    max = Math.max(tmap.get(key), max);
    tmap.put(key, max);
  }
  Map.Entry<Integer, Integer> entry = null;
  for (int i = 0; i < worker.length; i++) {
    entry = tmap.floorEntry(worker[i]);
    if (entry != null) {
      res += entry.getValue();
    }
  }
  return res;
}
```

## [Positions of Large Groups](https://leetcode.com/problems/positions-of-large-groups/description/)

```java
public List<List<Integer>> largeGroupPositions(String S) {
  int startIndex = 0;
  int endIndex = 0;
  int n = S.length();
  List<List<Integer>> res = new ArrayList<>();
  while (endIndex < n) {
    while (endIndex - startIndex >= 3 && S.charAt(endIndex) == S.charAt(endIndex)) {
      endIndex++;
    }
    if (endIndex - startIndex >= 3) {
      res.add(new ArrayList(Arrays.asList(startIndex, endIndex - 1);
    }
    startIndex = endIndex;
  }
  return res;
}
```

## [Max Area of Island](https://leetcode.com/problems/max-area-of-island/description/)

```java
public int maxAreaOfIsland(int[][] grid) {
  int maxArea = 0;
  for (int i = 0; i < grid.length; i++) {
    for (int j = 0; j < grid[0].length; j++) {
      if (grid[i][j] == 1) {
        maxArea = Math.max(maxArea, areaOfIsland(grid, j, j);
      }
    }
  }
  return maxArea;
}

private int areaOfIsland(int[][] grid, int i , int j) {
  if (i >= 0 && i < grid.length && j >= 0 && j < grid[0].length && grid[i][j] == 1) {
    grid[i][j] = 0;
    return 1 + areaOfIsland(grid, i + 1, j) + areaOfIsland(grid, i - 1, j) + areaOfIsland(grid, i, j + 1) + areaOfIsland(grid, i, j - 1);
  }
  return 0;
}
```

## [Single Num](https://leetcode.com/problems/single-number/description/)

```java
public int SingleNumber(int[] nums) {
  int result = 0;
  for (int i = 0; i < nums.length; i++) {
    result ^= nums[i];
  }
  return result;
}
```

## [Permutations](https://leetcode.com/problems/permutations/description/)

```java
public List<List<Integer>> permute(int[] nums) {
  List<List<Integer>> list = new ArrayList<>();
  backtrack(list, new ArrayList<>(), nums);
  return list;
}

private void backtrack(List<List<Integer>> list, List<Integer> tempList, int[] nums) {
  if (tempList.size() == nums.length) {
    list.add(new ArrayList(tempList));
  } else {
    for (int i = 0; i < nums.length; i++) {
      if (tempList.contains(nums[i])) {
        continue;
      }
      tempList.add(nums[i]);
      backtrack(list, tempList, nums);
      tempList.remove(tempList.size() - 1);
    }
  }
}
```

## [Advantage Shuffle](https://leetcode.com/problems/advantage-shuffle/description/)

```java
public int[] advantageCount(int[] A, int[] B) {
  Arrays.sort(A);
  int n = A.length;
  int[] res = new int[n];
  PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> b[0] - a[0]);
  for (int i = 0; i < n; i++) {
    pq.add(new int[]{B[i], i};
  }
  int low = 0;
  int high = n - 1;
  while (!pq.isEmpty()) {
    int[] cur = pq.poll();
    int idx = cur[1];
    int val = cur[0];
    if (A[high] > val) {
      res[idx] = A[high--];
    } else {
      res[idx] = A[low++];
    }
  }
  return res;
}
```
        
## [Monotone Increasing Digits](https://leetcode.com/problems/monotone-increasing-digits/description/)

```java
public int monotoneIncreasingDigits(int N) {
  if (N <= 9) {
    return N;
  }
  int mark = x.length;
  for (int i = x.length - 1; i > 0; i--) {
    if (x[i] < x[i - 1]) {
      mark = i - 1;
      x[i - 1]--;
    }
  }
  for (int i = mark + 1; i < x.length; i++) {
    x[i] = '9';
  }
  return Integer.parseInt(new String(x));
}
```


## [Remove Nth Node From End of List](https://leetcode.com/problems/remove-nth-node-from-end-of-list/description/)

```java
public ListNode removeNthFromEnd(ListNode head, int n) {
  ListNode start = new ListNode(0);
  ListNode fast = start;
  ListNode slow = start;
  slow.next = head;
  for (int i = 0; i <= n; i++) {
    fast = fast.next;
  }
  while (fast != null) {
    slow = slow.next;
    fast = fast.next;
  }
  slow.next = slow.next.next;
  return start.next;
}
```

## [Group Anagrams](https://leetcode.com/problems/group-anagrams/description/)

```java
public List<List<String>> groupAnagrams(String[] strs) {
  List<List<String>> result = new ArrayList<>();
  if (strs == null || strs.length == 0) {
    return result;
  }
  HashMap<String, List<String>> map = new HashMap<>();
  for (String s: strs) {
    char[] arr = s.toCharArray();
    Arrays.sort(arr);
    String str = String.valueOf(arr);
    if (!map.containsKey(str)) {
      map.put(str, new ArrayList<String>());
    }
    map.get(str).add(s);
  }
  return new ArrayList<List<String>>(map.values());
}
```

## [Course Schedule](https://leetcode.com/problems/course-schedule/description/)

```java
public boolean canFinish(int numCourses, int[][] prerequisities) {
  int[][] matrix = new int[numCourses][numCourses];
  int[] indegree = new int[numCourses];
  
  for (int i = 0; i < prerequisities.length; i++) {
    int ready = prerequisities[i][0];
    int pre = prerequisities[i][1];
    if (matrix[pre][ready] == 0) {
      indegree[ready]++;
    }
    matrix[pre][ready] = 1;
  }
  
  int count = 0;
  Queue<Integer> queue = new LinkedList();
  for (int i = 0; i < indegree.length; i++) {
    if (indegree[i] == 0) {
      queue.offer(i);
    }
  }
  while(!queue.isEmpty()) {
    int course = queue.poll();
    count++;
    for (int i = 0; i < numCourses; i++) {
      if (matrix[course][i] != 0) {
        if (--indegree[i] == 0) {
	  queue.offer(i);
	}
      }
    }
  }
  return count == numCourses;
}
```

## [Count Binary Substrings](https://leetcode.com/problems/count-binary-substrings/description/)

```java
public int countBinarySubstrings(String s) {
  int prevRunLength = 0;
  int curRunLength = 1;
  int res = 0;
  for (int i = 1; i < s.length(); i++) {
    if (s.charAt(i) == s.charAt(i -1)) {
      curRunLength++;
    } else {
      prevRunLength = curRunLength;
      curRunLength = 1;
    }
    if (prevRunLength >= curRunLength) {
      res++;
    }
  }
  return res;
}
```

## [Populating Next Right Pointers in Each Node II](https://leetcode.com/problems/populating-next-right-pointers-in-each-node-ii/description/)

```java
public void connect(TreeLinkNode root) {
  TreeLinkNode head = null;
  TreeLinkNode prev = null;
  TreeLinkNode cur = root;
  
  while (cur != null) {
    while (cur != null) {
      if (cur.left != null) {
        if (prev != null) {
	  prev.next = cur.left;
	} else {
	  head = cur.left;
	}
	prev = cur.left;
      }
      if (cur.right != null) {
        if (prev != null) {
	  prev.next = cur.right;
	} else {
	  head = cur.right;
	}
	prev = cur.right;
      }
      cur = cur.next;
    }
    cur = head;
    head = null;
    prev = null;
  }
}
```

## [Increasing Subsequences](https://leetcode.com/problems/increasing-subsequences/description/)

```java
public List<List<Integer>> findSubsequences(int[] nums) {
  Set<List<Integer>> res = new HashSet<List<Integer>>();
  List<Integer> holder = new ArrayList<Integer>();
  findSequence(res, holder, 0, nums);
  List result = new ArrayList(res);
  return result;
}

private void findSequence(Set<List<Integer>> res, List<Integer> holder, int index, int[] nums) {
  if (holder.size() >= 2) {
    res.add(new ArrayList(holder));
  }
  for (int i = index; i < nums.length; i++) {
    if (holder.size() == 0 || holder.get(holder.size() - 1) <= nums[i]) {
      holder.add(nums[i]);
      findSequence(res, holder, i + 1, nums);
      holder.remove(holder.size() -1);
    }
  }
}
```

## [Longest Mountain in Array](https://leetcode.com/problems/longest-mountain-in-array/description/)

```java
public int longestMountain(int[] A) {
  int n = A.length;
  int res = 0;
  int[] up = new int[n];
  int[] down = new int[n];
  for (int i = n - 2; i >= 0; i--) {
    if (A[i] > A[i + 1]) {
      down[i] = down[i + 1] + 1;
    }
  }
  for (int j = 0; i < n; j++) {
    if (j > 0 && A[j - 1] < A[j]) {
      up[j] = up[j - 1] + 1;
    }
    if (up[j] > 0 && down[j] > 0) {
      res = Math.max(res, up[j] + down[j] + 1);
    }
  }
  return res;
}
```

## [Binary Tree Right Side View](https://leetcode.com/problems/binary-tree-right-side-view/description/)

```java
public List<Integer> rightSideView(TreeNode root) {
  List<Integer> result = new ArrayList<Integer>();
  rightView(root, result, 0);
  return result;
}

private void rightView(TreeNode cur, List<Integer> result, int curDepth) {
  if (cur == null) {
    return;
  }
  if (curDepth == result.size()) {
    result.add(cur.val);
  }
  rightView(cur.right, result, curDepth + 1);
  rightView(cur.left, result, curDepth + 1);
}
```

## [Remove K Digits](https://leetcode.com/problems/remove-k-digits/description/)

```java
public String removeKdigits(String num, int k) {
  int digits = num.length() - k;
  char[] stk = new char[num.length()];
  int top = 0;
  for (int i = 0; i < num.length(); i++) {
    char c = num.charAt(i);
    while (top > 0 && stk[top - 1] > c && k > 0) {
      top--;
      k--;
    }
    stk[top++] = c;
  }
  int idx = 0;
  while (idx < digits && stk[idx] == '0') {
    idx++;
  }
  return idx == digits ? "0" : new String(stk, idx, digits -idx);
}
```

## [Number of Islands](https://leetcode.com/problems/number-of-islands/description/)

```java
private int m;
private int n;

public int numIslands(char[][] grid) {
  if (grid == null || grid.length == 0) {
    return 0;
  }
  int cnt = 0;
  n = grid.length;
  m = grid[0].length;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      if (grid[i][j] == '1') {
        cnt++;
	markDfs(grid, i, j);
      }
    }
  }
  return cnt;
}

private void arkDfs(char[][] grid, int i, int j) {
  if (i < 0 || j < 0 || i >= n || j >= m || grid[i][j] != '1') {
    return;
  }
  grid[i][j] = '0';
  markDfs(grid, i - 1, j);
  markDfs(grid, i + 1, j);
  markDfs(grid, i, j + 1);
  markDfs(grid, i, j - 1);
}
```

## [Judge Route Circle](https://leetcode.com/problems/judge-route-circle/description/) 

```java
public boolean judgeCircle(String moves) {
  int x = 0;
  int y = 0;
  for (char c: moves.toCharArray()) {
    if (c == 'R') {
	x++;
    } else if (c == 'L') {
	x--;
    } else if (c == 'U') {
	y++;
    } else if (c == 'D') {
	y--;
    }
  }
  return x == 0 && y == 0;
}
```

## [Maximum Product of Word Lengths](https://leetcode.com/problems/maximum-product-of-word-lengths/description/)

```java
public int maxProduct(String[] words) {
  if (words == null || words.length == 0) {
    return 0;
  }
  int len = words.length;
  int[] value = new int[len];
  for (int i = 0; i < len; i++) {
    String tmp = words[i];
    for (int j = 0; j < tmp.length(); j++) {
      value[i] |= 2 << (tmp.charAt(j) - 'a');
    }
  }
  int maxProduct = 0;
  for (int i = 0; i < len; i++) {
    for (int j = i + 1; j < len; j ++) {
      if ((value[i] & value[j]) == 0 && words[i].length() * words[j].length() > maxProduct) {
        maxProduct = words[i].length() * words[j].length();
      }
    }
  }
  return maxProduct;
}
```

## [Perfect Squares](https://leetcode.com/problems/perfect-squares/description/)

```java
public int numSquares(int n) {
  if (n <= 0) {
    return 0;
  }
  int[] dp = new int[n + 1];
  Arrays.fill(dp, Integer.MAX_VALUE);
  dp[0] = 1;
  for (int i = 1; i <=n; i++) {
    for (int j = 1; j * j <= i; j++) {
      dp[i] = Math.min(dp[i], dp[i -j * j] + 1);
    }
  }
  return dp[n];
}
```

## [Score After Flipping Matrix](https://leetcode.com/problems/score-after-flipping-matrix/description/)

```java
public int matrixScore(int[][] A) {
  int m = A.length;
  int n = A[0].length;
  int res = (1 << (n - 1)) * m;
  for (int j = 1; j < n; j ++) {
    int cur = 0;
    for (int i = 0; i < m; i++) {
      cur += A[i][j] == A[i][0] ? 1 : 0;
    }
    res += Math.max(cur, m - cur) * (1 << (n -j - 1));
  }
  return res;
}
```


## [2 Keys Keyboard](https://leetcode.com/problems/2-keys-keyboard/description/)

```java
public int minSteps(int n) {
  int res = 0;
  for (int i = 2; i <= n; i++) {
    while (n % i == 0) {
      res += i;
      n /= i;
    }
  }
  return res;
}
```

## [Majority Element](https://leetcode.com/problems/majority-element/description/)

```java
public int majorityElement(int[] nums) {
  int major = nums[0];
  int times = 1;
  for (int i = 1; i < nums.length; i++) {
    if (times == 0) {
      times++;
      major = nums[i];
    } else if (major == nums[i]) {
      times++;
    } else {
      times--;
    }
  }
  return major;
}
```


## [Maximum Product of Three Numbers](https://leetcode.com/problems/maximum-product-of-three-numbers/description/)

```java
public int maximumProduct(int[] nums) {
  int max1 = Integer.MIN_VALUE;
  int max2 = Integer.MIN_VALUE;
  int max3 = Integer.MIN_VALUE;
  int min1 = Integer.MIN_VALUE;
  int min2 = Integer.MIN_VALUE;
  for (int num: nums) {
    if (n > max1) {
      max3 = max2;
      max2 = max1; 
      max1 = n;
    } else if (n > max2) {
      max3 = max2;
      max2 = n;
    } else if (n > max3) {
      max3 = n;
    }
    
    if (n < min1) {
      min2 = min1;
      min1 = n;
    } else if (n < min2) {
      min2 = n;
    }
  }
  return Math.max(max1 * max2 * max3, max1 * min1 * min2);
}
```

## [Best Time to Buy and Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/description/)

```java
public int maxProfit(int[] prices) {
  int maxCur = 0;
  int maxSofar = 0;
  for (int i = 1; i < prices.length; i++) {
    maxCur = Math.max(0, maxCur += prices[i] - prices[i - 1]);
    maxSofar = Math.max(maxCur, maxSofar);
  }
  return maxSofar;
}
```

## [Target Sum](https://leetcode.com/problems/target-sum/description/)

```java
public int findTargetSumWays(int[] nums, int S) {
  int sum = 0;
  for (int num: nums) {
    sum += num;
  }
  return sum < S || (S + sum) % 2 != 0 ? 0 : subsetSum(nums, (sum + S) / 2);
}

private int subsetSum(int[] nums, int s) {
  int[] dp = new int[s + 1];
  dp[0] = 1;
  for (int num: nums) {
    for (int i = s; i >= num; i--) {
      dp[i] += dp[i - num];
    }
  }
  return dp[s];
}
```
    
## [Next Greater Element II](https://leetcode.com/problems/next-greater-element-ii/description/)

```java
public int[] nextGreaterElements(int[] nums) {
  int n = nums.length;
  int[] next = new int[n];
  Stack<Integer> s = new Stack<Integer>();
  Arrays.fill(next, -1);
  for(int i = 0; i < n * 2; i++) {
    int num = nums[i % n];
    while (!s.isEmpty() && nums[s.peek()] < num) {
      next[s.pop()] = num;
    }
    if (i < n) {
      s.push(i);
    }
  }
  return next;
}
```

## [Lexicographical Numbers](https://leetcode.com/problems/lexicographical-numbers/description/)

```java
public List<Integer> lexicalOrder(int n) {
  int cur = 1;
  List<Integer> list = new ArrayList<Integer>();
  for (int i = 1; i <= n; i++) {
    list.add(cur);
    if (cur * 10 <= n) {
      cur *= 10;
    } else if (cur % 10 != 9 && cur + 1 <= n) {
      cur++:
    } else {
      while ((cur /10) % 10 == 9) {
        cur /= 10;
      }
      cur = cur / 10 + 1;
    }
  }
  return list;
}
```

## [Maximum XOR of Two Numbers in an Array](https://leetcode.com/problems/maximum-xor-of-two-numbers-in-an-array/description/)

```java
public int findMaximumXOR(int[] nums) {
  int max = 0;
  int mask = 0;
  for (int i = 31; i >= 0; i--) {
    mask = mask | (1 << i);
    Set<Integer> set = new HashSet<Integer>();
    for (int num: nums) {
      set.add(num & mask);
    }
    int tmp = max | (1 << i);
    for (int prefix: set) {
      if (set.contains(tmp ^ prefix)) {
        max = tmp;
	break;
      }
    }
  }
  return max;
}
``` 

## [Jewels and Stones](https://leetcode.com/problems/jewels-and-stones/description/)

```java
public int numJewelsInStones(String J, String S) {
  int res = 0;
  HashSet set = new HashSet();
  for (char j: J.toCharArray()) {
    set.add(j);
  }
  for (char s: S.toCharArray()) {
    if (set.contains(s)) {
      res++;
    }
  }
  return res;
}
```

## [Self Dividing Numbers](https://leetcode.com/problems/self-dividing-numbers/description/)

```java
public List<Integer> selfDividingNumbers(int left, int right) {
  List<Integer> list = new ArrayList<Integer>();
  for (int i = left; i <= right; i++) {
    int j = i;
    for (; j > 0; j /= 10) {
      if (j % 10 == 0 || i % (j % 10) != 0) {
        break;
      }
    }
    if (j == 0) {
      list.add(i);
    }
  }
  return list;
}
```

## [Array Nesting](https://leetcode.com/problems/array-nesting/description/)

```java
public int arrayNesting(int[] nums) 
  int maxsize = 0;
  for (int i = 0; i < nums.length; i++) {
    int size = 0;
    for (int k = i; nums[k] >= 0; size++) {
      int ak = nums[k];
      nums[k] = -1;
      k = ak;
    }
    maxsize = Math.max(size, maxsize);
  }
  return maxsize;
}
```

## [Task Scheduler](https://leetcode.com/problems/task-scheduler/description/)

```java
public int leastInterval(char[] tasks, int n) {
  int[] c = new int[26];
  for (char t: tasks) {
    c[t - 'A']++;
  }
  Arrays.sort(c);
  int i = 25;
  while (i >= 0 && c[i] == c[25]) {
    i--;
  }
  return Math.max(tasks.length, (c[25] - 1) * (n - 1) + 25 - i);
}
```

## [Sum Root to Leaf Numbers](https://leetcode.com/problems/sum-root-to-leaf-numbers/description/)

```java
public int sumNumbers(TreeNode root) {
  return sum(root, 0);
}

private int sum(TreeNode root, int s) {
  if (root == null) {
    return 0;
  }
  if (root.left == null && root.right == null) {
    return s * 10 + root.val;
  }
  return sum(root.left, s * 10 + root.val) + sum(root.right, s * 10 + root.val);
}
```

## [Populating Next Right Pointers in Each Node](https://leetcode.com/problems/populating-next-right-pointers-in-each-node/description/)

```java
public void connect(TreeLinkNode root) {
  TreeLinkNode pre = root;
  while (pre != null) {
    TreeLinkNode cur = pre;
    while (cur != null) {
      if (cur.left != null) {
        cur.left.next = cur.right;
      }
      if (cur.right != null && cur.next != null) {
        cur.right.next = cur.next.left;
      }
      cur = cur.next;
    }
    pre = pre.left;
  }
}
```

## [Valid Perfect Square](https://leetcode.com/problems/valid-perfect-square/description/)

```java
public boolean isPerfectSquare(int num) {
  long x = num;
  while (x * x > num) {
    x = (x + num / x) >> 1;
  }
  return x * x == num;
}
```

## [4Sum II](https://leetcode.com/problems/4sum-ii/description/)

```java
public int fourSumCount(int[] A, int[] B, int[] C, int[] D) {
  HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();
  
  for (int i = 0; i < C.length; i++) {
    for (int j = 0; j < D.length; j++) {
      int sum = C[i] + D[j];
      map.put(sum, map.getOrDefault(sum, 0) + 1);
    }
  }
  
  int res = 0;
  for (int i = 0; i < A.length; i++) {
    for (int j = 0; j < B.length; j++) {
      res += map.getOrDefault(-1 * (A[i] + B[j]), 0);
    }
  }
  return res;
}
```

## [Two Sum IV - Input is a BST](https://leetcode.com/problems/two-sum-iv-input-is-a-bst/description/)

```java
public boolean findTarget(TreeNode root, int k) {
  HashSet<Integer> set = new HashSet<Integer>();
  return dfs(root, set, k);
}

private boolean dfs(TreeNode root, HashSet<Integer> set, int k) {
  if (root == null) {
    return false;
  }
  if (set.contains(k - root.val)) {
    return true;
  }
  set.add(root.val);
  return dfs(root.left, set, k) || dfs(root.right, set, k);
}
```

## [Coin Change 2](https://leetcode.com/problems/coin-change-2/description/)

```java
public int change(int amount, int[] coins) {
  int[] dp = new int[amount + 1];
  dp[0] = 1;
  for (int coin: coins) {
    for (int i = coin; i <= amount; i++) {
      dp[i] += dp[i - coin];
    }
  }
  return dp[amount];
}
```

## [Binary Tree Maximum Path Sum](https://leetcode.com/problems/binary-tree-maximum-path-sum/description/)

```java
int maxVal;
public int maxPathSum(TreeNode root) {
  maxVal = Integer.MIN_VALUE;
  maxPathDown(root);
  return maxVal;
}

private int maxPathDown(TreeNode root) {
  if (root == null) {
    return 0;
  }
  int left = Math.max(0, maxPathDown(root.left));
  int right = Math.max(0, maxPathDown(root.right));
  maxVal = Math.max(maxVal, left + right + root.val);
  return Math.max(left, right) + root.val;
}
```


## [Letter Case Permutation](https://leetcode.com/problems/letter-case-permutation/description/)

```java
public List<String> letterCasePermutation(String S) {
  if (S == null) {
    return new LinkedList<>();
  }
  Queue<String> q = new LinkedList<>();
  q.offer(S);
  for (int i = 0; i < S.length(); i++) {
    if (Character.isDigit(S.charAt(i))) {
      continue;
    }
    int n = q.size();
    for (int j = 0; j < n; j++) {
      String cur = q.poll();
      char[] chs = cur.toCharArray();
      chs[i] = Character.toUpperCase(chs[i]);
      q.offer(String.valueOf(chs));
      chs[i] = Character.toLowerCase(chs[i]);
      q.offer(String.valueOf(chs));
    }
  }
  return new LinkedList<>(q);
}
```

## [Ugly Number](https://leetcode.com/problems/ugly-number/description/)

```java
public boolean isUgly(int num) {
  if (num <= 0) {
    return false;
  }
  for (int i = 2; i < 6; i++) {
    while (num % i == 0) {
      num /= i;
    }
  }
  return num == 1;
}
```

## [Spiral Matrix II](https://leetcode.com/problems/spiral-matrix-ii/description/)

```java
public int[][] generateMatrix(int n) {
  int[][] matrix = new int[n][n];
  if (n == 0) {
    return matrix;
  }
  int colStart = 0;
  int colEnd = n - 1;
  int rowStart = 0;
  int rowEnd = n - 1;
  int num = 1;
  while (rowStart <= rowEnd && colStart <= colEnd) {
    for (int i = colStart; i <= colEnd; i++) {
      matrix[rowStart][i] = num++;
    }
    rowStart++;
    
    for (int i = rowStart; i <= rowEnd; i++) {
      matrix[i][colEnd] = num++;
    }
    colEnd--;
    
    for (int i = colEnd; i >= colStart; i--) {
      if (rowStart <= rowEnd) {
        matrix[rowEnd][i] = num++;
      }
    }
    rowEnd--;
    
    for (int i = rowEnd; i >= rowStart; i--) {
      if (colStart <= colEnd) {
        matrix[i][colStart] = num++;
      }
    }
    colStart++;
  }
  return matrix;
}
```

## [Find Largest Value in Each Tree Row](https://leetcode.com/problems/find-largest-value-in-each-tree-row/description/)

```java
public List<Integer> largestValues(TreeNode root) {
  List<Integer> result = new ArrayList<Integer>();
  if (root == null) {
    return result;
  }
  Queue<TreeNode> q = new LinkedList<TreeNode>();
  q.offer(root);
  while (!q.isEmpty()) {
    int levelNum = q.size();
    int maxVal = q.peek().val;
    for (int i = 0; i < levelNum; i++) {
      if (q.peek().left != null) {
        q.offer(q.peek().left);
      }
      if (q.peek().right != null) {
        q.offer(q.peek().right);
      }
      int val = q.poll().val;
      if (val > maxVal) {
        maxVal = val;
      }
    }
    result.add(maxVal);
  }
  return result;
}
```

## [Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/description/)

```java
public List<List<Integer>> levelOrder(TreeNode root) {
  List<List<Integer>> result = new LinkedList<List<Integer>>();
  if (root == null) {
    return result;
  }
  Queue<TreeNode> q = new LinkedList<TreeNode>();
  q.offer(root);
  while (!q.isEmpty()) {
    int levelNum = q.size();
    List<Integer> subList = new LinkedList<Integer>();
    for (int i = 0; i < levelNum; i++) {
      if (q.peek().left != null) {
        q.offer(q.peek().left);
      }
      if (q.peek().right != null) {
        q.offer(q.peek().right);
      }
      subList.add(q.poll().val);
    }
    result.add(subList);
  }
  return result;
}
```
      

## [Isomorphic Strings](https://leetcode.com/problems/isomorphic-strings/description/)

```java
public boolean isIsomorphic(String s, String t) {
  int[] m1 = new int[256];
  int[] m2 = new int[256];
  int n = s.length();
  for (int i = 0; i < n; i++) {
    if (m1[s.charAt[i] != m2[t.charAt(i)) {
      return false;
    }
    m1[s.charAt(i)] = i + 1;
    m2[t.charAt(i)] = i + 1;
  }
  return true;
}
```

## [Search in Rotated Sorted Array](https://leetcode.com/problems/search-in-rotated-sorted-array/description/)

```java
public int search(int[] nums, int target) {
  if (nums == null || nums.length == 0) {
    return -1;
  }
  int low = 0;
  int high = nums.length - 1;
  while (low < high) {
    int mid = (low + high) / 2;
    if (nums[mid] == target) {
      return mid;
    }
    if (nums[low] <= nums[mid]) {
      if (target >= nums[low] && target < nums[mid]) {
         high = mid - 1;
      } else {
         low = mid + 1;
      }
    } else {
      if (target > nums[mid] && target <= nums[high]) {
        low = mid + 1;
      } else {
        high = mid - 1;
      }
    }
  }
  return nums[low] == target ? low : -1;
}
```

## [Unique Paths II](https://leetcode.com/problems/unique-paths-ii/description/)

```java
public int uniquePathsWithObstacles(int[][] obstacleGrid) { 
  int width = obstacleGrid[0].length;
  int[] dp = new int[width];
  dp[0] = 1;
  for (int[] row : obstacleGrid) {
    for (int i = 0; i < width; i++) {
      if (row[i] == 1) {
        dp[i] = 0;
      } else if (i > 0) {
        dp[i] += dp[i - 1];
      }
    }
  }
  return dp[width - 1];
}
 ```

## [Construct Binary Tree from Inorder and Postorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/description/)

```java
public TreeNode buildTree(int[] inorder, int[] postorder) {
  if (inorder == null || postorder == null || inorder.length != postorder.length) {
    return null;
  }
  HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();
  for (int i = 0; i < inorder.length; i ++) {
    map.put(inorder[i], i);
  }
  return buildPostIn(inorder, 0, inorder.length - 1, postorder, 0, postorder.length - 1, map);
}

private TreeNode buildPostIn(int[] inorder, int is, int ie, int[] postorder, int ps, int pe, HashMap<Integer, Integer> map) {
  if (ps > pe || is > ie) {
    return null;
  }
  TreeNode root = new TreeNode(postorder[pe]);
  int index = map.get(postorder[pe]);
  TreeNode left = buildPostIn(inorder, is, index - 1, postorder, ps, ps + index - is - 1, map);
  TreeNode right = buildPostIn(inorder, index + 1, ie, postorder, ps + index - is, pe - 1, map);
  root.left = left;
  root.right = right;
  return root;
}
```

## [Minimum Distance Between BST Nodes](https://leetcode.com/problems/minimum-distance-between-bst-nodes/description/)

```java
Integer res = Integer.MAX_VALUE;
Integer pre = null;
public int minDiffInBST(TreeNode root) {
  if (root.left != null) {
    minDiffInBST(root.left);
  }
  if (pre != null) {
    res = Math.min(res, root.val - pre);
  }
  pre = root.val;
  if (root.right != null) {
    minDiffInBST(root.right);
  }
  return res;
}
```


## [House Robber II](https://leetcode.com/problems/house-robber-ii/description/)

```java
public int rob(int[] nums) {
  if (nums.length == 1) {
    return nums[0];
  }
  return Math.max(rob(nums, 0, nums.length - 2), rob(nums, 1, nums.length - 1));
}

private int rob(int[] nums, int low, int high) {
  int include = 0;
  int exclude = 0;
  for (int i = low; i <= high; i++) {
    int j = include;
    int e = exclude;
    include = e + nums[i];
    exclude = Math.max(j, e);
  }
  return Math.max(include, exclude);
}
```

## [Balanced Binary Tree](https://leetcode.com/problems/balanced-binary-tree/description/)

```java
public boolean isBalanced(TreeNode root) {
  if (root == null) {
    return true;
  }
  int left = depth(root.left);
  int right = depth(root.right);
  return Math.abs(left - right) <= 1 && isBalanced(root.left) && isBalanced(root.right);
}

private int depth(TreeNode root) {
  if (root == null) {
    return 0;
  }
  return Math.max(depth(root.left), depth(root.right)) + 1;
}
```

## [Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/description/)

```java
public int lengthOfLongestSubstring(String s) {
  int len = s.length();
  if (len == 0) {
   return 0;
  }
  HashMap<Character, Integer> map = new HashMap<Character, Integer>();
  int max = 0;
  for (int i = 0, j = 0; i < len; i++) {
    Character c = s.charAt(i);
    if (map.containsKey(c)) {
      j = Math.max(j, map.get(c) + 1);
    }
    map.put(c, i);
    max = Math.max(max, i - j + 1);
  }
  return max;
}
```
 

## [Total Hamming Distance](https://leetcode.com/problems/total-hamming-distance/description/)

```java
public int totalHammingDistance(int[] nums) {
  int total = 0;
  int n = nums.length;
  for (int i = 0; i < 32; i++) {
    int bitCount = 0;
    for (int j = 0; j < n; j++) {
      bitCount += (nums[j] >> i) & 1;
    }
    total += bitCount * (n - totalCount);
  }
  return total;
}
```

## [Delete Node in a Linked List](https://leetcode.com/problems/delete-node-in-a-linked-list/description/)

```java
public void deleteNode(ListNode node) {
  node.val = node.next.val;
  node.next = node.next.next;
}
```

## [Spiral Matrix](https://leetcode.com/problems/spiral-matrix/description/)

```java
public List<Integer> spiralOrder(int[][] matrix) {
  List<Integer> result = new ArrayList<Integer>();
  if (matrix == null || matrix.length == 0) {
    return result;
  }
  int rowBegin = 0;
  int rowEnd = matrix.length - 1;
  int colBegin = 0;
  int colEnd = matrix[0].length - 1;
  while (rowBegin <= rowEnd && colBegin <= colEnd) {
    for (int i = colBegin; i <= colEnd; i++) {
      result.add(matrix[rowBegin][i]);
    }
    rowBegin++;
    
    for (int i = rowBegin; i <= rowEnd; i++) {
      result.add(matrix[i][colEnd]);
    }
    colEnd--;
    
    if (rowBegin <= rowEnd) {
      for (int i = colEnd; i >= colBegin; i--) {
        result.add(matrix[rowEnd][i]);
      }
    }
    rowEnd--;
    
    if (colBegin <= colEnd) {
      for (int i = rowEnd; i >= rowBegin; i--) {
        result.add(matrix[i][colBegin]);
      }
    }
    colBegin++;
  }
  return result;
}
```

## [Flatten Binary Tree to Linked List](https://leetcode.com/problems/flatten-binary-tree-to-linked-list/description/)

```java
TreeNode pre = null;
public void flatten(TreeNode root) {
  if (root == null) {
    return;
  }
  flatten(root.right);
  flatten(root.left);
  root.right = pre;
  root.left = null;
  pre = root;
}
```

## [Longest Repeating Character Replacement](https://leetcode.com/problems/longest-repeating-character-replacement/description/)

```java
public int characterReplacement(String s, int k) {
  int len = s.length();
  int[] count = new int[26];
  int maxCount = 0;
  int maxLen = 0;
  int start = 0;
  for (int end = 0; end < len; end++) {
    maxCount = Math.max(maxCount, ++count[s.charAt(end) - 'A']);
    while (end - start + 1 - maxCount > k) {
      count[s.charAt(start) - 'A']--;
      start++;
    }
    maxLen = Math.max(maxLen, end - start + 1);
  }
  return maxLen;
}
```

## [Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/description/)

```java
public ListNode reverseList(ListNode head) {
  ListNode pre = null;
  while(head != null) {
    ListNode next = head.next;
    head.next = pre;
    pre = head;
    head = next;
  }
  return pre;
}
```

## [Longest Increasing Subsequence](https://leetcode.com/problems/longest-increasing-subsequence/description/)

```java
public int lengthOfLTS(int[] nums) {
  int[] dp = new int[nums.length];
  int len = 0;
  
  for (int x:nums) {
    int i = Arrays.binarySearch(dp, 0, len, x);
    if (i < 0) {
      i = -(i + 1);
    }
    dp[i] = x;
    if (i == len) {
      len++;
    }
  }
  return len;
}
```

## [Most Frequent Subtree Sum](https://leetcode.com/problems/most-frequent-subtree-sum/description/)

```java
public int[] findFrequentTreeSum(TreeNode root) {
  HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();
  sum(root, map);
  List<Integer> list = new ArrayList<Integer>();
  int max = 0;
  for (Map.Entry<Integer, Integer> entry: map.entrySet()) {
    int val = map.getValue();
    if (val >= max) {
      if (val > max) {
        list.clear();
	max = val;
      }
      list.add(val);
    }
  }
  int[] res = new int[list.size()];
  for (int i = 0; i < list.size(); i++) {
    res[i] = list.get(i);
  }
  return res;
}

private int sum(TreeNode root, HashMap<Integer, Integer> map) {
  if (root == null) {
    return 0;
  }
  int left = sum(root.left, map);
  int right = sum(root.right, map);
  int sum = left + right + root.val;
  map.put(sum, map.getOrDefault(sum, 0) + 1);
  return sum;
}
```

## [Min Cost Climbing Stairs](https://leetcode.com/problems/min-cost-climbing-stairs/description/)

```java
public int minCostClimbingStairs(int[] cost) {
  int n = cost.length;
  int[] dp = new int[n];
  dp[0] = cost[0];
  dp[1] = cost[1];
  for (int i = 2; i < n; i++) {
    dp[i] = cost[i] + Math.min(dp[i - 1], dp[i - 2]);
  }
  return Math.min(dp[n - 2], dp[n - 1]);
}
```

## [Diameter of Binary Tree](https://leetcode.com/problems/diameter-of-binary-tree/description/)
```
int max = 0;

public int diameterOfBinaryTree(TreeNode root) {
  maxDepth(root);
  return max;
}

private int maxDepth(TreeNode root) {
  if (root == null) {
    return 0;
  }
  int left = maxDepth(root.left);
  int right = maxDepth(root.right);
  max = Math.max(max, left + right);
  return Math.max(left, right) + 1;
}
```
## [Happy Number](https://leetcode.com/problems/happy-number/description/)

```java
public boolean isHappy(int n) {
  Set<Integer> loop = new HashSet<Integer>();
  int squareSum, remain;
  while (loop.add(n)) {
    squareSum = 0;
    while (n > 0) {
      remain = n % 10;
      squareSum += remain * remain;
      n /= 10;
    }
    if (squareSum == 1) {
      return true;
    } else {
      n = squareSum;
    }
  }
  return false;
}
```
      

## [Custom Sort String](https://leetcode.com/problems/custom-sort-string/description/)

```java
public String customSortString(String S, String T) {
  int[] count = new int[26];
  for (char c: T.toCharArray()) {
    ++count[c - 'a'];
  }
  StringBuilder sb = new StringBuilder();
  for (char c: S.toCharArray()) {
    while(cnt[c - 'a'] -- > 0) {
      sb.append(c);
    }
  }
  for (char c = 'a'; c <= 'z'; c++) {
    while(cnt[c - 'a']-- > 0) {
    sb.append(c);
    }
  }
  return sb.toString();
}
```


## [Maximum Product Subarray](https://leetcode.com/problems/maximum-product-subarray/description/)

```java
public int maxProduct(int[] nums) {
  int r = nums[0];
  for (int i = 1, imin = r, imax = r; i < nums.length; i++) {
    if (nums[i] < 0) {
      int tmp = imin;
      imin = imax;
      imax = tmp;
    }
    imin = Math.min(nums[i], nums[i] * imin);
    imax = Math.max(nums[i], nums[i] * imax);
    r = Math.max(r, imax);
  }
  return r;
}
```

## [Reverse Linked List II](https://leetcode.com/problems/reverse-linked-list-ii/description/)

```java
public ListNode reverseBetween(ListNode head, int m, int n) {
  if (head == null) {
    return null;
  }
  ListNode dummy = new ListNode(0);
  dummy.next = head;
  ListNode pre = dummy;
  for (int i = 0; i < m - 1; i++) {
    pre = pre.next;
  }
  ListNode start = pre.next;
  ListNode then = start.next;
  for (int i = 0; i < n - m; i++) {
    start.next = then.next;
    then.next = pre.next;
    pre.next = then;
    then = start.next;
  }
  return dummy.next;
}
```

## [Rotate String](https://leetcode.com/problems/rotate-string/description/)

```java
public boolean rotateSteing(String A, String B) {
  return A.length() == B.length() && (A + A).contains(B);
}
```

## [Number of Subarrays with Bounded Maximum](https://leetcode.com/problems/number-of-subarrays-with-bounded-maximum/description/)

```java
public int numSubarrayBoundedMax(int[] A, int L, int R) {
  int res = 0;
  for (int i = 0; i M A,length; i++) {
    if (A[i] > R) {
      continue;
    }
    int max = Integer.MIN_VALUE;
    for (int j = i; j < A.length; j++) {
      max = Math.max(max, A[j]);
      if (max > R) {
        break;
      }
      if (max >= L) {
        res++;
      }
    }
  }
  return res;
}
```

## [Longest Univalue Path](https://leetcode.com/problems/longest-univalue-path/description/)

```java
public int longestUnivaluePath(TreeNode root) {
  int[] res = new int[1];
  if (root != null) {
    dfs(root, res);
  }
  return res[0];
}

private int dfs (TreeNode root, int[] res) {
  int l = root.left == null ? 0 : dfs(root.left, res);
  int r = root.right == null ? 0 : dfs(root.right, res);
  int resl = root.left != null && root.left.val == root.val ? l + 1 : 0;
  int resr = root.right != null && root.right.val == root.val ? r + 1 : 0;
  res[0] = Math.max(res[0], resl + resr);
  return Math.max(resl, resr);
}
```

## [Permutations II](https://leetcode.com/problems/permutations-ii/description/)

```java
public List<List<Integer>> permuteUnique(int[] nums) {
  List<List<Integer>> result = new ArrayList<List<Integer>>();
  if (nums == null || nums.length == 0) {
    return result;
  }
  boolean[] used = new boolean[nums.length];
  Arrays.sort(nums);
  List<Integer> list = new ArrayList<Integer>();
  dfs(nums, used, list, result);
  return result;
}

private void dfs (int[] nums, boolean[] used, List<Integer> list, List<List<Integer>> result) {
  if (nums.length == list.size()) {
    result.add(new ArrayList<Integer>(list));
    return;
  }
  for (int i = 0; i < nums.length; i++) {
    if (used[i]) {
      continue;
    }
    if (i > 0 && nums[i] == nums[i - 1] && !used[i - 1]) {
      continue;
    }
    used[i] = true;
    list.add(nums[i]);
    dfs(nums, used, list, result);
    used[i] = false;
    list.remove(list.size() - 1);
  }
}
```
  

## [Median of Two Sorted Arrays](https://leetcode.com/problems/median-of-two-sorted-arrays/description/)

```java 
public double findMedianSortedArrays(int[] nums1, int[] nums2) {
  int m = nums1.length;
  int n = nums2.length;
  if (m < n) {
    return findMedianSortedArrays(nums2, nums1);
  }
  int low = 0;
  int high = n * 2;
  while (low <= high) {
    int mid2 = (low + high) / 2;
    int mid1 = m + n - mid2;
    double L1 = mid1 == 0 ? Integer.MIN_VALUE : nums1[(mid1 - 1) / 2];
    double L2 = mid2 == 0 ? Integer.MIN_VALUE : nums2[(mid2 - 1) / 2];
    douvle R1 = mid1 == m * 2 ? Integer.MAX_VALUE : nums1[mid1 / 2];
    double R2 = mid2 == n * 2 ? Integer.MAX_VALUE : nums2[mid2 / 2];
    
    if (L1 > R2) {
      low = mid2 + 1;
    } else if (L2 > R1) {
      high = mid2 - 1;
    } else {
      return (Math.max(L1, L2) + Math.Min(R1, R2)) / 2;
    }
  }
  return -1;
}
```

## [Average of Levels in Binary Tree](https://leetcode.com/problems/average-of-levels-in-binary-tree/description/)

```java
public List<Double> averageOfLevels(TreeNode root) {
  List<Double> result = new ArrayList<Double>();
  Queue<TreeNode> q = new LinkedList<TreeNode>();
  if (root == null) {
    return result;
  }
  q.add(root);
  while(!q.isEmpty()) {
    int n = q.size();
    double sum = 0;
    for (int i = 0; i < n; i++) {
      TreeNode node = q.poll();
      sum += node.val;
      if (node.left != null) {
        q.offer(node.left);
      }
      if (node.right != null) {
        q.offer(node.right);
      }
    }
    result.add(sum / n);
  }
  return result;
}
```

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
	
