# Leetcode

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
            } else {
                left++;
            }
        }
        return indices;
    }
}
```

## 14. Minimum size subarray sum

```java
public class Solution {
    public int minSubArrayLen(int s, int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int i = 0, j = 0, sum = 0, min = Integer.MAX_VALUE;
        while (j < nums.length) {
            sum += nums[j++];
            while (sum >= s) {
                min = Math.min(min, j-i);
                sum -= nums[i++];
            }
        }
        return min == Integer.MAX_VALUE ? 0 : min;
    }
}
```

## 15.Combination sum

```java
public class Solution {
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        Arrays.sort(candidates);
        List<List<Integer>> result = new ArrayList<List<Integer>>();
        getResult(result, new ArrayList<Integer>(), candidates, target, 0);
        return result;
    }
    
    private void getResult(List<List<Integer>> result, List<Integer> cur, int[] candidates, int target, int start) {
        if (target > 0) {
            for (int i = start; i < candidates.length && target >= candidates[i]; i ++) {
                cur.add(candidates[i]);
                getResult(result, cur, candidates, target - candidates[i], i);
                cur.remove(cur.size() - 1);
            }
        } else if (target == 0) {
            result.add(new ArrayList(cur));
        } else {
            return;
        }
    }
}
```

## 16.Combination sum II

```java
public class Solution {
    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        Arrays.sort(candidates);
        List<List<Integer>> result = new ArrayList();
        getResult(result, new ArrayList(), candidates,target, 0);
        return result;
    }
    
    private void getResult(List<List<Integer>> result, List<Integer> cur, int[] candidates, int target, int start) {
        if (target < 0) {
            return;
        } else if (target == 0) {
            result.add(new ArrayList(cur));
        } else {
            for (int i = start; i < candidates.length; i ++) {
                if (i > start && candidates[i] == candidates[i-1]) {
                    continue;
                }
                cur.add(candidates[i]);
                getResult(result, cur, candidates, target - candidates[i], i + 1);
                cur.remove(cur.size() - 1);
            }
        }
    }
}
```

## 17.Combination sum III

```java
public class Solution {
    public List<List<Integer>> combinationSum3(int k, int n) {
        List<List<Integer>> ans = new ArrayList();
        combination(ans, new ArrayList<Integer>(), k, 1 ,n);
        return ans;
    }
    
    private void combination(List<List<Integer>> ans, List<Integer> combo, int k, int start, int n) {
        if (combo.size() == k && n == 0) {
            List<Integer> li = new ArrayList(combo);
            ans.add(li);
            return;
        }
        
        for (int i = start; i <= 9; i++) {
            combo.add(i);
            combination(ans, combo, k, i+1, n-i);
            combo.remove(combo.size()-1);
        }
    }
}
```

## 18.Combination sum IV

```java
public class Solution {
    Map<Integer, Integer> map = new HashMap<Integer, Integer>();
    public int combinationSum4(int[] nums, int target) {
        if (nums == null || target < 0 || nums.length == 0) {
            return 0;
        }
        if (target == 0) {
            return 1;
        }
        int count = 0;
        if (map.containsKey(target)) {
            return map.get(target);
        }
        for (int num : nums) {
            count += combinationSum4(nums, target - num);
        }
        map.put(target, count);
        return count;
    }
}
```

## 19.Integer break

```java
public class Solution {
    public int integerBreak(int n) {
        if(n==2) return 1;
        if(n==3) return 2;
        int product = 1;
        while(n>4){
            product*=3;
            n-=3;
        }
        product*=n;
        
        return product;
    }
}
```

## 20.Product of array except self

```java
public class Solution {
    public int[] productExceptSelf(int[] nums) {
        int n = nums.length;
        int[] res = new int[n];
        res[0] = 1;
        for (int i = 1;i < n;i++) {
            res[i] = res[i-1] * nums[i-1];
        }
        int right = 1;
        for (int j = n -1;j >= 0;j--) {
            res[j] *= right;
            right *= nums[j];
        }
        return res;
    }
}
```

## 21.[Find the duplicate number](https://leetcode.com/problems/find-the-duplicate-number/)

```java
public class Solution {
    public int findDuplicate(int[] nums) {
        if (nums == null || nums.length < 2) {
            return 0;
        }
        int len = nums.length;
        int low = 1;
        int high = len - 1;
        int mid;
        int count = 0;
        while (low < high) {
            mid = low + (high - low)/2;
            for (int x : nums) {
                if (x <= mid) {
                    count ++;
                }
            }
            if (count > mid) {
                high = mid;
            } else {
                low = mid + 1;
            }
            count = 0;
        }
        return low;
        
    }
}
```

## 22. BFS tree serialize and deserialize

```java
import java.util.*;
public class Main {
    public static void main(String[] args) {
	// write your code here
        TreeNode a = new TreeNode("5");
        TreeNode b = new TreeNode("2");
        TreeNode c = new TreeNode("3");
        TreeNode d = new TreeNode("4");
        TreeNode e = new TreeNode("1");
        a.left = b;
        a.right = c;
        c.left = d;
        c.right = e;
        String str = "5,2,3,#,#,1,4";
        TreeNode node = deserialize(str);
        serialize(node);

    }

    public static class TreeNode {
        String val;
        TreeNode left;
        TreeNode right;
        TreeNode (String x) {
            val = x;
        }
    }

    public static String serialize(TreeNode root) {
        String result = "";
        Deque<TreeNode> q = new ArrayDeque<>();
        q.add(root);
        TreeNode node = null;
        while (!q.isEmpty()) {
            node = q.remove();
            result += node.val;
            if (node.left != null) {
                q.add(node.left);
            }
            if (node.right != null) {
                q.add(node.right);
            }
        }
        return result;
    }

    public static TreeNode deserialize(String data) {
        Deque<String> nodes = new LinkedList<>();
        nodes.addAll(Arrays.asList(data.split(",")));
        return buildTree(nodes);
    }

    private static TreeNode buildTree(Deque<String> nodes) {
        String val = nodes.remove();
        if (val.equals("#")) {
            return null;
        } else {
            TreeNode node = new TreeNode(val);
            node.left = buildTree(nodes);
            node.right = buildTree(nodes);
            return node;
        }
    }
}
```

## 23. Judges whether same of two trees

```java
public bool isSame(TreeNode root1, TreeNode root2) {
  if (root1 == null && root2 == null) {
    return true;
  }
  if(root1 != null || root2 != null) {
    return false;
  }
  return (root1.val == root2.val) && isSame(root1.left, root2.left) && isSame(root1.right, root2.right);
}
```

## 24.Obtain the mirror tree

```java
public void obtainMirrorTree(TreeNode root) {
  if (root == null) {
    return;
  }
  if (root.left == null && root.right == null) {
    return;
  }
  TreeNode temp = root.left;
  root.left = root.right;
  root.right = temp;
  obtainMirrorTree(root.left);
  obtainMirrorTree(root.right);
}
```

## 25.Lowest common ancestor of a binary search tree

```java
public TreeNode search(TreeNode root, TreeNode p, TreeNode q) {
  if (root.val > p.val && root.val > q.val) {
    return search(root.left, p, q);
  } else if (root.val < p.val && root.val < q.val) {
    return search(root.right, p, q);
  } else {
    return root;
  }
}
```

## 26. Lowest common ancestor of a binary tree

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
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == p || root == q || root == null) {
            return root;
        }
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        if (left != null && right != null) {
            return root;
        }
        if (left != null) {
            return left;
        } else {
            return right;
        }
    }
}
```

## 27. Nth smallest element in a BST

```java
  public int kthSmallest(TreeNode root, int k) {
        int count = countNodes(root.left);
        if (k <= count) {
            return kthSmallest(root.left, k);
        } else if (k > count + 1) {
            return kthSmallest(root.right, k-1-count); // 1 is counted as current node
        }
        
        return root.val;
    }
    
    public int countNodes(TreeNode n) {
        if (n == null) return 0;
        
        return 1 + countNodes(n.left) + countNodes(n.right);
    }
```

## 28.[Find all numbers disappeared in an array](https://leetcode.com/problems/find-all-numbers-disappeared-in-an-array/)

```java
public class Solution {
    public List<Integer> findDisappearedNumbers(int[] nums) {
        List<Integer> result = new ArrayList<Integer>();
        int len = nums.length;
        for (int i = 0; i <len; i++) {
            int val = Math.abs(nums[i]) - 1;
            if (nums[val] > 0) {
                nums[val] = -nums[val];
            }
        }
        for (int j = 0; j < len; j++) {
            if (nums[j] > 0) {
                result.add(j + 1);
            }
        }
        return result;
    }
}
```

